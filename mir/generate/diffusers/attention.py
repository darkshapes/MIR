# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


# def gen_attention_processors(mir_db: MIRDatabase): # upstream not quite ready for this yet
#     from diffusers.models.attention_processor import AttentionProcessor

#     mir_data
#     for series, comp_name in mir_data.items():
#     id_segment = series.split(".")
#     for compatibility in comp_name:
#         dbug(id_segment)
#         try:
#             mir_db.add(
#                 mir_entry(
#                     domain=id_segment[0],
#                     arch=id_segment[1],
#                     series=id_segment[2],
#                     comp=compatibility,
#                     **mir_data[series][compatibility],
#                 ),
#             )
#         except IndexError as error_log:
#             nfo(f"Failed to create series: {series}  compatibility: {comp_name}  ")
#             dbug(error_log)

