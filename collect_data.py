from pathlib import Path
from langchain_community.document_loaders import WikipediaLoader
from openai import OpenAI
import csv
from tqdm import tqdm
import time



if __name__ == "__main__":
    out_dir = Path("input")
    out_dir.mkdir(exist_ok=True)

    client = OpenAI()

    prompt_list = []
    prompt_csv = Path("human survey prompts.csv")
    with open(prompt_csv, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_list.append(row["Prompt"])

    titles_searched = []
    for i, prompt in tqdm(enumerate(prompt_list)):
        prompt_dir = out_dir / str(i).zfill(3)
        prompt_dir.mkdir(exist_ok=True)

        completion = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"For a given prompt \"{prompt}\", identify relevant entities that are directly related to subjects appeared in the prompt (e.g., specific objects, groups, or concepts) and suitable to search in Wikipedia to help me better understand the appearance of those subjects. Return the response with the entities separated by '/'."
                }
            ]
        )

        entities = completion.choices[0].message.content
        entities = entities.split("/")
        if len(entities) > 3:
            entities = entities[:3]
        entities.insert(0, prompt)

        for ent in entities:
            docs = WikipediaLoader(query=ent, load_max_docs=3, doc_content_chars_max=10000).load()
            for doc in docs:
                title = doc.metadata['title']
                if title in titles_searched:
                    continue
                else:
                    titles_searched.append(title)
                with open(prompt_dir / f"{title}.txt", 'w') as f:
                    f.write(doc.page_content)
            
        time.sleep(2)
