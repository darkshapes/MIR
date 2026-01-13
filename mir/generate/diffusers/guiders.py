# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


# def gen_guiders(mir_db: MIRDatabase):  # upstream not quite ready for this yet
#     from nnll.metadata.helpers import snake_caseify
#     from diffusers.guider import GuiderType

#     guider_type = GuiderType
#     for comp_name in guider_type.items():
#         class_obj = comp_name.__name__
#         mir_data = {"pkg": {0: {"diffusers": class_obj}}}
#         try:
#             mir_db.add(
#                 mir_entry(
#                     domain="ops",
#                     arch="noise_prediction",
#                     series="guider",
#                     comp=snake_caseify(class_obj),
#                     **mir_data,
#                 ),
#             )
#         except IndexError as error_log:
#             nfo(f"Failed to create compatibility: {class_obj}")
#             dbug(error_log)


# (
#     "info.unet",
#     "stable-cascade",
#     {
#         "combined": {
#             "pkg": {
#                 0: {  # decoder=decoder_unet
#                     "precision": "ops.precision.bfloat.B16",
#                     "generation": {
#                         "negative_prompt": "",
#                         "num_inference_steps": 20,
#                         "guidance_scale": 4.0,
#                         "num_images_per_prompt": 1,
#                         "width": 1024,
#                         "height": 1024,
#                     },
#                 },
#                 "pkg_alt": {
#                     0: {
#                         "diffusers": {
#                             "StableCascadeCombinedPipeline": {
#                                 "negative_prompt": "",
#                                 "num_inference_steps": 10,
#                                 "prior_num_inference_steps": 20,
#                                 "prior_guidance_scale": 3.0,
#                             }
#                         },
#                     }
#                 },
#             }
#         }
#     },
# ),

