import json,yaml

def create_train_file(digit_file,train_output_file):
    

    with open(digit_file, 'r') as infile, open(train_output_file, 'w') as outfile:
        for line in infile:
            record = json.loads(line.strip())
            record['assistant'] = record.pop('response')
            record['system'] = ''
            record['user'] = record.pop('instruction')
            outfile.write(json.dumps(record) + '\n')

# def get_user_and_system(instruction):
#         split_delimiter = '\n'
#         user_start_string = "text: "
#         split_values = instruction.split(split_delimiter)
#         if split_values[0].startswith(user_start_string):
#             user = split_values[0]
#             system = split_delimiter.join(split_values[1:])
#         elif split_values[1].startswith(user_start_string):
#             system = split_values[0] + split_delimiter
#             user = split_delimiter.join(split_values[1:])
#         else:
#             raise ValueError("could not split to user and system: {instruction}")
#         return user,system

def create_test_file(yaml_file,test_output_file):
    with open(yaml_file, 'r') as f:
        yaml_content = yaml.safe_load(f)
        yaml_content = yaml_content.get("seed_examples", {})
    with open(test_output_file, 'w') as f:
        if isinstance(yaml_content, list):
            for entry in yaml_content:
                entry['user'] = entry.pop('question')
                entry['system'] = ''
                entry['assistant'] = entry.pop('answer')
                f.write(json.dumps(entry) + '\n')

if __name__=="__main__":
    input_file = 'ilab/sdg/watson_emotion_classes_first_sdg.jsonl'
    train_file = 'ilab/sdg/train_watson_emotion.jsonl'
    create_train_file(digit_file=input_file,train_output_file=train_file)
    yaml_file = 'ilab/sdg/watson_emotion_classes_first.yaml'
    test_file = 'ilab/sdg/test_watson_emotion.jsonl'
    create_test_file(yaml_file=yaml_file,test_output_file=test_file)