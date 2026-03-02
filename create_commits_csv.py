import json
import csv

# Load the JSON file
with open(r"Repo data\kafka_source_dataset.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create CSV file
output_file = "kafka_commits.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['commit_id', 'pr_id', 'label'])
    
    # Write commit data
    for commit_hash, commit_info in data.items():
        pr_number = commit_info.get('pr_number')
        writer.writerow([commit_hash, pr_number, 0])

print(f"CSV file created: {output_file}")
print(f"Total commits processed: {len(data)}")
