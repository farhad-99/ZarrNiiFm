
from zarrnii import ZarrNii, ZarrNiiAtlas
from zarrnii.plugins.segmentation import LocalOtsuSegmentation,VesselFM

img = ZarrNii.from_ome_zarr('/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3/bids/sub-AS164F5/micr/sub-AS164F5_sample-brain_acq-imaris4x_SPIM.ome.zarr', channel_labels=['CD31'],level=5, downsample_near_isotropic=True)

atlas = ZarrNiiAtlas.from_files('/tmp/ZarrNii/ZarrNii/data/lightsheet/sub-AS134F3_sample-brain_acq-imaris4x_seg-all_from-ABAv3_level-5_desc-deform_dseg.nii.gz',
                        '/tmp/ZarrNii/ZarrNii/data/lightsheet/seg-all_tpl-ABAv3_dseg.tsv')

cropped = img.crop_with_bounding_box(*atlas.get_region_bounding_box(regex='Hipp'),ras_coords=True)
cropped.to_nifti('/tmp/ZarrNii/tmp_results/cropped.nii')

plugin = VesselFM()
print(plugin)
segmented = img.segment(plugin)
segmented.to_nifti('/tmp/ZarrNii/tmp_results/reees.nii')
img.to_nifti('/tmp/ZarrNii/tmp_results/img.nii')
