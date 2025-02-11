import logging
import openai


def get_chat_response(prompt, model='gpt-4o', max_token=256, retry=5):
    messages = [
        {'role': 'user', 'content': prompt},
    ]
    for i in range(retry):
        try:
            completion = openai.chat.completions.create(model=model,
                                                        messages=messages,
                                                        temperature=0.5 * i,
                                                        max_tokens=max_token)
            prediction = completion.choices[0].message.content.strip()
            if prediction != '' and prediction is not None:
                return prediction
            else:
                continue
        except Exception as e:
            logging.error(e)
    return ''
