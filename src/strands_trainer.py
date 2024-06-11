from pytorch3d.io import IO
import torch
import torch.nn.functional as F

from src.hair_networks.optimizable_textured_strands import OptimizableTexturedStrands
from src.hair_networks.strands_renderer import Renderer
from src.losses.sdf_chamfer import SdfChamfer
from src.losses.one_way_chamfer import chamfer_distance

from src.utils.util import freeze_gradients


class StrandsTrainer:
    def __init__(self, config, run_model=None, device=None, save_dir=None) -> None:
        
        self.device = device
        
        params_to_train = []
        self.strands = OptimizableTexturedStrands(**config['textured_strands'], diffusion_cfg=config['diffusion_prior']).to(self.device)
        params_to_train += list(self.strands.parameters())

        self.strands_render = None
        if config['render']['use_render']:
            print('Create rasterizer!')
            self.strands_render = Renderer(config['render'], save_dir=save_dir).to(self.device)
            params_to_train += list(self.strands_render.parameters())
        else:
            print(f'No Rendering Used!')

        self.starting_rendering_iter = config['general']['starting_rendering_iter']  

        self.optimizer = torch.optim.Adam(params_to_train, config['general']['lr'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config['general']['milestones'], gamma=config['general']['gamma'])
        self.loss_factors = config['loss_factors']

        self.sdfchamfer = SdfChamfer(**config['sdf_chamfer'])
        self.run_model = run_model
        self.strands_origins = None
        self.orientation = IO().load_pointcloud(config['general']['orientation_path'], device=self.device)
    
    def get_latent_codes(self):
        """
        :return: latent codes which were optimized during training
        """
        latent_codes = torch.nn.Embedding.from_pretrained(
            torch.load('./implicit-hair-data/data/nphm/039/latent_best.ckpt', map_location='cpu')['weight']
        )
        latent_codes.to(self.device)
        return latent_codes

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.strands.load_state_dict(state_dict['strands'])
        if self.strands_render is not None:
            self.strands_render.load_state_dict(state_dict['strands_render'])


    def save_weights(self, path):
        state_dict = {
            'strands': self.strands.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if self.strands_render is not None:
            state_dict['strands_render'] = self.strands_render.state_dict()
        torch.save(state_dict, path)

    def sort_by_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(tensor, p=2, dim=1)
        idx = torch.argsort(norm, stable=True)
        return tensor[idx]

    def train_step(self, model=None, it=0, raster_dict=None):
        losses = {}

        self.strands_origins, z_geom, z_app, dif_dict = self.strands(it=it)
        strand_len = self.strands_origins.shape[1]
        
        with freeze_gradients(model):
            out = self.run_model(model, self.strands_origins.view(-1, 3))
            # grads = model.gradient(self.strands_origins.view((-1, 3)))
            # if it == 0:
            #     evaluate_hairsdf_on_grid(model.decoder, self.device, 256, 'sdf_output_nsh.obj')

        # Calculate origin loss
        # sdf = out[..., 0].view(-1, strand_len)
        sdf = out.view(-1, strand_len)
        sdf_inside = torch.relu(sdf[:, 1:])
        losses['volume'] = F.mse_loss(sdf_inside, torch.zeros_like(sdf_inside))

        # Calculate orientation loss
        prim_dir = self.strands_origins[:, 1:] - self.strands_origins[:, :-1] # [N_strands, strand_len-1, 3]

        # n_strands = prim_dir.shape[0]
        # pred_dir = F.grid_sample(self.orientation[None, None, ...].type(torch.float32), self.grid, align_corners=True, mode='bilinear')
        # pred_dir = pred_dir.squeeze()[:, :-1]

        # # Calculate orientations only near the visible outer surface
        dist = self.sdfchamfer.points2face(self.sdfchamfer.mesh_outer_hair_remeshed, self.strands_origins[:, :-1, :].reshape(-1, 3)) 
        filtered_idx = torch.nonzero(dist[0] <= 0.001).reshape(-1)
        # losses['orient'] = (1 - torch.abs(torch.cosine_similarity(prim_dir.reshape(-1, 3)[filtered_idx], self.orientation[filtered_idx], dim=-1))).mean()
        _, loss_orient = chamfer_distance(
            x = self.strands_origins[:, :-1, :].reshape(-1, 3)[filtered_idx].unsqueeze(0),
            x_normals = prim_dir.reshape(-1, 3)[filtered_idx].unsqueeze(0),
            y = self.orientation.points_packed().to(prim_dir.dtype).unsqueeze(0), 
            y_normals = self.orientation.normals_packed().to(prim_dir.dtype).unsqueeze(0),
            batch_reduction=None
        )
        losses['orient'] = loss_orient.mean(dim=1)
        self.loss_orient = loss_orient.detach().clone().reshape(-1, strand_len - 1).numpy(force=True)

        # Calculate chamfer on visible outer surface
        if self.sdfchamfer is not None:
            losses['chamfer'] = self.sdfchamfer.calc_chamfer(self.strands_origins[:, 1:, :].reshape(-1, 3)[None])

        # Calculate photometric losses
        if self.strands_render and it > self.starting_rendering_iter:
            raster_dict = self.strands_render( 
                                        strands_origins=self.strands_origins,
                                        z_app=z_app,
                                        raster_dict=raster_dict,
                                        iter=it,
                                         )

            raster_loss = self.strands_render.calculate_losses(raster_dict, it)

            if 'silh' in raster_loss.keys():
                losses['raster_silh'] = raster_loss['silh']
            if 'alpha_prediction' in raster_loss.keys():
                losses['raster_alpha'] = raster_loss['alpha_prediction']
            losses['raster_l1'] = raster_loss['l1']
        
        # Calculate diffusion loss
        if len(dif_dict) > 0:
            losses['L_diff'] = dif_dict['L_diff']

        self.optimizer.zero_grad()

        total_loss = sum(loss * float(self.loss_factors[name]) for name, loss in losses.items())
        total_loss.backward()

        for param in self.optimizer.param_groups[0]['params']:
            if param.grad is not None and param.grad.isnan().any():
                self.optimizer.zero_grad()
                print('NaN during backprop was found, skipping iteration...')
                return losses
            
        self.optimizer.step()
        self.scheduler.step()

        return losses
