# Used https://github.com/ReagentX/imessage-exporter to extract ios messages
# imessage-exporter -f html -c disabled -a iOS -o raw

import os
import pathlib
import re
from lxml import html as HTMLParser

cwd_path = pathlib.Path(__file__).parent
raw_data_path = cwd_path / 'raw'
data_path = cwd_path / 'processed'

for filename in os.listdir(raw_data_path):
    if not filename.endswith('.html'): continue
    with open(raw_data_path / filename, 'r') as file:
        try:
            html = HTMLParser.parse(file)
        except:
            print(f'error reading file {filename}, ignored')
            continue

    elements = html.xpath("//body/div[contains(@class, 'message')] \
                          /div[contains(@class, 'sent') or contains(@class, 'received')]")
    msgs = []

    for e in elements:
        msg_content = e.xpath("./div[contains(@class, 'message_part')]/span")
        if not msg_content or len(msg_content) == 0: continue
        msg = msg_content[0].text_content()
        if re.findall(r'[\u4e00-\u9fff]', msg): continue # exclude chinese characters
        # if re.findall(r'[^a-zA-z0-9]', msg): continue # exclude non-english characters

        sender = e.xpath("./p/span[contains(@class, 'sender')]/text()")
        if not sender or len(sender) == 0: continue
        category = 'Sent' if sender[0] == "Me" else 'Recieved'

        msgs.append(f'##{category}\n{msg}\n\n')

    if len(msgs) != 0:
        with open(data_path / filename.replace('.html', '.txt'), 'w') as file:
            file.write(''.join(msgs))