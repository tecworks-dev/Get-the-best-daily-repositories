"""

convert anyword into llava like format


"""

import sys
import json


TYPE = "ego"
TYPE = "webvid"
TYPE = "videochat2"


def convert_json(input_json, json_file_path):
    data = input_json["data_list"]
    print(f"processing data list: {len(data)}")
    output_data = []
    for item in data:
        if "annotations" in item.keys():

            new_item = {
                # "id": f"ego_video_{video_path}",
                "id": item["img_name"],
                # "video": f"split_videos/{video_path}",
                "image": f"images/{item['img_name']}",
            }

            new_item["conversations"] = []

            ocr_text = []
            all_lans = []
            for i, QA_data in enumerate(item["annotations"]):
                ocr_text.append(QA_data["text"])
                all_lans.append(QA_data["language"])

            should_use = True
            if "laion" in json_file_path and len(all_lans) > 0:
                if any(lang != "Latin" for lang in all_lans):
                    should_use = False
                    # print(item)
                else:
                    print(item)
            # if len(new_item["conversations"]) > 5:
            #     # depart conversations into 2 parts
            #     for i in range(0, len(new_item["conversations"]), 10):
            #         data_dict_i = {}
            #         data_dict_i["id"] = new_item["id"] + f"_{i//10}"
            #         data_dict_i["video"] = new_item["video"]
            #         data_dict_i["conversations"] = new_item["conversations"][i : i + 10]
            #         if i != 0:
            #             data_dict_i["conversations"][0][
            #                 "value"
            #             ] = f"<video>\n{data_dict_i['conversations'][0]['value']}"
            #         output_data.append(data_dict_i)
            # else:
            if should_use:
                new_item["conversations"].append(
                    {
                        "from": "human",
                        "value": "<image>\nPlease provide precise OCR result of the image.",
                    }
                )
                new_item["conversations"].append(
                    {"from": "gpt", "value": "\n".join(ocr_text)}
                )
                output_data.append(new_item)

    return output_data


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    try:
        with open(json_file_path, "r") as file:
            input_json = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} does not exist.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file {json_file_path} is not a valid JSON file.")
        sys.exit(1)

    converted_data = convert_json(input_json, json_file_path)
    print(f"All {len(converted_data)} samples")
    # file_path = "ego_video.json"
    file_path = f"{json_file_path[:-5]}_ocr.json"
    with open(file_path, "w") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
