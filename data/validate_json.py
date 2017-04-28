old_file_name = "snli_1.0_dev.jsonl"
new_file_name = "snli_1.0_dev.json"

# Add comma at end of every line in old (invalid) JSON.
with open(old_file_name, 'r') as istr:
    with open(new_file_name, 'w') as ostr:
        for line in istr:
            line = line.rstrip('\n') + ','
            print(line, file=ostr)

# Append "[" to start and "]" to end of new valid json. Manually remove the last comma. 
with open(new_file_name, 'r') as original: data = original.read()
with open(new_file_name, 'w') as modified: modified.write("[\n" + data + "\n]")
