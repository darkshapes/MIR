# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import os
import re

# def tag_mlx_model(repo_path: str, class_name: str, addendum: dict) -> tuple[str]:
#     dev_series, dev_comp = make_mir_tag("black-forest-labs/FLUX.1-dev")
#     schnell_series, schnell_comp = make_mir_tag("black-forest-labs/FLUX.1-schnell")
#     series, comp = make_mir_tag(repo_path)
#     if class_name == "Flux1":
#         mir_prefix = "info.dit"
#         base_series = dev_series
#         mir_comp = series
#         return mir_prefix, base_series, {base_comp: addendum}


def mlx_repo_capture(base_repo: str = "mlx-community"):
    try:
        import mlx_audio  # type: ignore
    except ImportError:
        return {}
    result = {}
    result_2 = {}
    folder_path_named: str = os.path.dirname(mlx_audio.__file__)
    for root, dir, file_names in os.walk(folder_path_named):
        for file in file_names:
            if file.endswith((".py", ".html", ".md", ".ts")):
                with open(os.path.join(root, file), "r") as open_file:
                    content = open_file.read()
                    if "mlx-community/" in content:
                        matches = re.findall(base_repo + r'/(.*?)"', content)
                        for match in matches:
                            result[match] = f"{base_repo}/{match}"
                            previous_data = content[content.index(match) - 75 : content.index(match)].replace(base_repo, "")
                            class_match = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)
                            if class_match:
                                result_2[match] = {f"{base_repo}/{match}": [*class_match]}
                            else:
                                if os.path.basename(root) in ["tts", "sts"]:
                                    folder_name = match.partition("-")[0]
                                    file_path = os.path.join(root, "models", folder_name, folder_name + ".py")
                                    if os.path.exists(file_path):
                                        with open(file_path, "r") as model_file:
                                            read_data = model_file.read()  # type: ignore  # noqa
                                            class_match = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)

    return result_2


# def mlx_repo_capture(base_repo: str = "mlx-community"):
#     import os
#     import re
#     import mlx_audio

#     result = {}
#     result_2 = {}
#     folder_path_named: str = os.path.dirname(mlx_audio.__file__)
#     for root, _, file_names in os.walk(folder_path_named):
#         for file in file_names:
#             if file.endswith((".py", ".html", ".md", ".ts")):
#                 with open(os.path.join(root, file), "r") as open_file:
#                     content = open_file.read()
#                     if "mlx-community/" in content:
#                         matches = re.findall(base_repo + r'/(.*?)"', content)
#                         for match in matches:
#                             print(file)
#                             result[match] = f"{base_repo}/{match}"
#                             previous_data = content[content.index(match) - 75 : content.index(match)].replace(base_repo, "")
#                             matches = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)
#                             if matches:
#                                 result_2[match] = {f"{base_repo}/{match}": [*matches]}
#                             else:
#                                 result_2[match] = {f"{base_repo}/{match}": None}
#     return result_2


# def mlx_audio_scrape(base_repo: str = "mlx-community"):
#     import os
#     import re
#     import mlx_audio

#     result = {}
#     result_2 = {}
#     folder_path_named: str = os.path.dirname(mlx_audio.__file__)
#     for root, _, file_names in os.walk(folder_path_named):
#         for file in file_names:
#             if file.endswith((".py",)):
#                 with open(os.path.join(root, file), "r") as open_file:
#                     content = open_file.read()
#                     if "mlx-community/" in content:
#                         matches = re.findall(base_repo + r'/(.*?)"', content)
#                         for match in matches:
#                             result[match] = f"{base_repo}/{match}"
#                             previous_data = content[content.index(match) - 75 : content.index(match)].replace(base_repo, "")
#                             matches = re.findall(r"(\w+)\.from_pretrained", previous_data, re.MULTILINE)
#                             if len(matches) > 1:
#                                 result_2[match] = {f"{base_repo}/{match}": [*matches]}
#                             else:
#                                 if "nn.Module" in content:
#                                     previous_data = content[content.rindex("nn.Module") - 50 : content.rindex("nn.Module")]
#                                     matches = re.search(r"(\w+)\.", previous_data, re.MULTILINE)
#                                     result_2[match] = {f"{base_repo}/{match}": [*matches]}
#     return result_2
