import json
import re

with open('message_1.json', 'r') as f:
    data = json.load(f)
    messages = data['messages']

# Write a regex that  replaces Ã¶ with ö, Ã¤ with ä, Ã¼ with ü, ÃŸ with ß, and Ã„ with Ä
def replace_characters(text):
    replacements = {
        'Ã¶': 'ö',
        'Ã¤': 'ä',
        'Ã¼': 'ü',
        'ÃŸ': 'ß',
        'Ã„': 'Ä',
        "Ã¥": "å",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

with open('fbinput.txt', 'w') as f:
    for message in messages:
        if not "content" in message:
            continue
        author = replace_characters(message['sender_name'])
        content = replace_characters(message['content'])
        pattern = r'[^a-zA-Z0-9åäöÅÄÖ]+'
        content = re.sub(pattern, ' ', content)
        f.write(f"{author}: \n")
        f.write(f"{content}\n")
        f.write("\n")
