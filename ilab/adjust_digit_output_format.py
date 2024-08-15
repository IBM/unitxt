import json,yaml,argparse

def create_train_file(digit_file,train_output_file):
    

    with open(digit_file, 'r') as infile, open(train_output_file, 'w') as outfile:
        for line in infile:
            record = json.loads(line.strip())
            record['assistant'] = record.pop('response')
            record['system'] = ''
            record['user'] = record.pop('instruction')
            outfile.write(json.dumps(record) + '\n')

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

    parser = argparse.ArgumentParser(description='evaluate dataset against ilab model and save results')
    
    parser.add_argument('--sdg_file', type=str, required=True, help='path of file created by sdg')
    parser.add_argument('--yaml_file', type=str, required=True, help='path of yaml file')
    parser.add_argument('--train_file',required=True, type=str,help='path of train output file')
    parser.add_argument('--test_file',required=True, type=str,help='path of test output file')
    args = parser.parse_args()
    
    create_train_file(digit_file=args.sdg_file, train_output_file=args.train_file)
    create_test_file(yaml_file=args.yaml_file, test_output_file=args.test_file)