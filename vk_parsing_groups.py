import requests
import csv

def take_posts():
    token = '4d428b6e4d428b6e4d428b6e274e666c6c44d424d428b6e2a0091d81da35c2108a6468e'
    version = '5.199'
    domain = 'princessss_kr'
    posts = []
    response = requests.get('https://api.vk.com/method/wall.get',
                            params={
                                'access_token': token,
                                'v': version,
                                'domain': domain
                            })
    data = response.json()['response']['items']
    posts.extend(data)  # Добавляем объекты (словарей) в список
    return posts

def file_writer(all_posts):
    with open('test.csv', 'w', newline='', encoding='utf-8') as file:
        a_pen = csv.writer(file)
        a_pen.writerow(('body', 'url'))
        for post in all_posts:
            try:
                if 'attachments' in post and post['attachments'][0]['type'] == 'photo':
                    img_url = post['attachments'][0]['photo']['sizes'][-1]['url']
                else:
                    img_url = 'pass'
            except (KeyError, IndexError):
                img_url = 'pass'
            a_pen.writerow((post['text'], img_url))  # Безопасное извлечение текста

all_posts = take_posts()
file_writer(all_posts)
