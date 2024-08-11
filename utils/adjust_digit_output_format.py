import json

def adjust_digit_sdg_format(input_file,output_file):
    split_delimiter = '\n'

    renames = {'instruction':'system', 'response':'assistant'}

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            record = json.loads(line.strip())
            for field in renames:
                if field in record:
                    record[renames[field]] = record.pop(field)
            split_values = record['system'].split(split_delimiter)
            record['user'] = split_values[0] + split_delimiter
            record['system'] = split_delimiter.join(split_values[1:])
            outfile.write(json.dumps(record) + '\n')

if __name__=="__main__":
    input_file = 'sdg/digit_watson_emotion.jsonl'
    output_file = 'sdg/sdg_watson_emotion.jsonl'
    adjust_digit_sdg_format(input_file=input_file,output_file=output_file)