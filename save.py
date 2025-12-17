
from zarrnii import ZarrNii, ZarrNiiAtlas
from zarrnii.plugins.segmentation import LocalOtsuSegmentation,VesselFM
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="inference", version_base="1.3.2")
def main(cfg: DictConfig):

    # Load image
    chunk_size = (1,500,500,500)
    img = ZarrNii.from_ome_zarr('/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3/bids/sub-AS164F5/micr/sub-AS164F5_sample-brain_acq-imaris4x_SPIM.ome.zarr', channel_labels=['CD31'],level=3, downsample_near_isotropic=True, chunks=chunk_size, rechunk=True)

    
    atlas = ZarrNiiAtlas.from_files('/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch1/derivatives/spimquant_aae813e/sub-AS134F3/micr/sub-AS134F3_sample-brain_acq-imaris4x_seg-all_from-ABAv3_level-5_desc-deform_dseg.nii.gz',
                            '/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch1/derivatives/spimquant_aae813e/tpl-ABAv3/seg-all_tpl-ABAv3_dseg.tsv')
    # Crop using atlas region
    cropped = img.crop_with_bounding_box(*atlas.get_region_bounding_box(regex='right Agranular Insular Area'),ras_coords=True)

    cropped.to_nifti('/tmp/ZarrNii/tmp_results/cropped.nii')

    # Create plugin from Hydra config
   #print(plugin)
    plugin = VesselFM(cfg=cfg)
    
    segmented = cropped.segment(plugin)
    print('cropped shape:', cropped.data.shape)
    segmented.to_nifti('/tmp/ZarrNii/tmp_results/reees.nii')
    img.to_nifti('/tmp/ZarrNii/tmp_results/img.nii')
    #print(cfg.device)
if __name__ == "__main__":
    main()