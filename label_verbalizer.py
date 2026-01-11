import json
from pathlib import Path

def get_label_category_information(label: str, verbalization_data: dict) -> str:
        """
        Retrieves category information for a given label from the verbalization data.
        :param label: The label to retrieve category information for.
        :return: A string containing the category information.
        """
        if not verbalization_data:
            raise ValueError("Verbalization data is not defined. Please set the verbalization_data attribute before calling this method.")

        if label not in verbalization_data:
            raise NotImplementedError(f"Label {label} not found in verbalization data.")

        categories = {'instance of': [], 'part of': [], 'country': [], 'occupation': [], 'subclass of': []}
        data = verbalization_data[label].get('wikidata_properties', [])

        for key, value in data:
            if key in categories:
                categories[key].append(value)

        category_info = ""
        for key, values in categories.items():
            if values:
                if category_info != "":
                    category_info += " ; "
                category_info += f"{key}: {', '.join(values)}"

        return category_info

if __name__ == "__main__":
    verbalizations_path = "./verbalizations_structured.json"
    verbalizations_dict = json.load(Path(verbalizations_path).open("r", encoding="utf-8"))
    print(verbalizations_dict["Apple_Inc."].get("wikidata_properties", {}))
    print(get_label_category_information("Apple_Inc.", verbalizations_dict))
    print(f"Loaded verbalizations for {len(verbalizations_dict)} items.")