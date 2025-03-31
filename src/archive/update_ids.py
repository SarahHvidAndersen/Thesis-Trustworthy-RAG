import os
import json

def update_ids_in_file(filepath, course_prefix):
    """
    Loads a JSON file, updates each chunk's "chunk_id" so that it starts with the course prefix,
    and writes the updated data back to the file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updated = False
    for chunk in data:
        current_id = chunk.get("chunk_id", "")
        if not current_id.startswith(f"{course_prefix}_"):
            # Prepend the course prefix.
            chunk["chunk_id"] = f"{course_prefix}_{current_id}"
            updated = True

    if updated:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Updated IDs in {filepath}")
    else:
        print(f"No update needed for {filepath}")

def main():
    # List of courses
    courses = ['Human_computer_interaction', 'Natural_language_processing', 
               'Adv_cog_neuroscience', 'Adv_cognitive_modelling', 'Data_science', 'Decision_making']
    
    base_dir = "processed_syllabi"
    for course in courses:
        # Adjust file name
        filepath = os.path.join(base_dir, course, "processed_chunks_with_embeddings.json")
        if os.path.exists(filepath):
            update_ids_in_file(filepath, course)
        else:
            print(f"File not found for course '{course}': {filepath}")

if __name__ == "__main__":
    main()
