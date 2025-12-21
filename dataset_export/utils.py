import uuid


def generate_class_folder_name(names_class: list[str]):
    combined = '-'.join(names_class)
    base_name = combined[:10] if len(combined) > 10 else combined

    folder_name = f'dt-{base_name}-{uuid.uuid4()}'
    return folder_name
