import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from stop_words import get_stop_words
import numpy as np
import time

# Константы
TOKEN = '4d428b6e4d428b6e4d428b6e274e666c6c44d424d428b6e2a0091d81da35c2108a6468e'  # Токен для доступа к API VK
VERSION = '5.199'  # Версия API VK

def safe_request(url, params, retries=3, delay=0.5):
    """
    Выполняет запрос к API VK с учетом повторных попыток в случае ошибок.

    :param url: URL API метода
    :param params: Параметры запроса
    :param retries: Количество попыток
    :param delay: Задержка между попытками
    :return: Ответ API в формате JSON
    """
    for attempt in range(retries):
        try:
            # Отправка GET-запроса
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Обработка ошибки "Слишком много запросов"
            if 'error' in data and data['error']['error_code'] == 6:
                print("Слишком много запросов. Повторная попытка...")
                time.sleep(delay * (attempt + 1))
                continue

            return data
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса: {e}")
            time.sleep(delay * (attempt + 1))

    # Возвращаем сообщение об ошибке после превышения количества попыток
    return {"error": {"error_msg": "Превышено количество попыток.", "error_code": 6}}

# Функция для получения ID пользователя по имени или screen_name

def get_user_id(user_name):
    """
    Получает числовой ID пользователя по его screen_name или короткому имени.

    :param user_name: Имя пользователя или screen_name
    :return: Числовой ID пользователя
    """
    params = {
        'access_token': TOKEN,
        'v': VERSION,
        'user_ids': user_name
    }
    data = safe_request('https://api.vk.com/method/users.get', params)
    if 'response' in data:
        user_id = data['response'][0]['id']
        return user_id
    else:
        print(f"Ошибка API: {data['error']['error_msg']} (Код: {data['error']['error_code']})")
    return None


def take_user_posts(user_id):
    """
    Получает тексты всех постов пользователя с его стены, включая репосты.

    :param user_id: ID пользователя VK
    :return: Список текстов постов
    """
    posts = []
    offset = 0
    while True:
        params = {
            'access_token': TOKEN,
            'v': VERSION,
            'owner_id': user_id,
            'count': 100,
            'offset': offset
        }
        data = safe_request('https://api.vk.com/method/wall.get', params)

        # Проверка на ошибки API
        if 'error' in data:
            print(f"Ошибка API: {data['error']['error_msg']} (Код: {data['error']['error_code']})")
            break

        items = data.get('response', {}).get('items', [])

        # Если больше нет постов, выходим из цикла
        if not items:
            break

        for post in items:
            text = post.get('text', '').strip()
            
            # Если текст пустой, пытаемся извлечь данные из copy_history (репосты)
            if not text and 'copy_history' in post:
                text = ' '.join(
                    copy.get('text', '').strip() 
                    for copy in post['copy_history'] 
                    if copy.get('text', '').strip()
                )

            # Добавляем текст поста в список, если он не пустой
            if text:
                posts.append(text)

        offset += 100  # Смещение для следующей выборки
        time.sleep(0.5)  # Задержка для предотвращения блокировки

    return posts


# Функция для получения подписок пользователя (групп)

def get_user_subscriptions(user_id):
    """
    Получает список названий групп, на которые подписан пользователь.

    :param user_id: ID пользователя VK
    :return: Список названий групп
    """
    group_ids = []
    offset = 0
    while True:
        params = {
            'access_token': TOKEN,
            'v': VERSION,
            'user_id': user_id,
            'count': 200,
            'offset': offset
        }
        data = safe_request('https://api.vk.com/method/users.getSubscriptions', params)

        # Проверка на ошибки API
        if 'error' in data:
            print(f"Ошибка API: {data['error']['error_msg']} (Код: {data['error']['error_code']})")
            break

        response = data.get('response', {})
        group_items = response.get('groups', {}).get('items', [])

        # Если больше нет групп или достигнут лимит, выходим
        if not group_items or offset > 1000:
            break

        group_ids.extend(group_items)
        offset += 200
        time.sleep(0.5)  # Задержка для предотвращения блокировки

    # Получение названий групп по их ID
    groups = []
    for i in range(0, len(group_ids), 500):  # Обработка ID батчами по 500
        batch_ids = group_ids[i:i + 500]
        params = {
            'access_token': TOKEN,
            'v': VERSION,
            'group_ids': ','.join(map(str, batch_ids)),
            'fields': 'name,type'
        }
        data = safe_request('https://api.vk.com/method/groups.getById', params)
        if 'error' in data:
            print(f"Ошибка API при получении данных о группах: {data['error']['error_msg']} (Код: {data['error']['error_code']})")
            continue

        group_info = data.get('response', {}).get('groups', [])
        for group in group_info:
            group_name = group.get('name')
            if group_name:
                groups.append(group_name)

    return groups

# Функция для анализа текстов с использованием TF-IDF

def analyze_texts(texts, vectorizer=None):
    """
    Выполняет обработку текстов и вычисляет их TF-IDF представление.

    :param texts: Список текстов
    :param vectorizer: Используемый векторизатор (если есть)
    :return: TF-IDF матрица и векторизатор
    """
    russian_stop_words = get_stop_words('russian')  # Получаем список русских стоп-слов

    # Убираем пустые строки
    texts = [text for text in texts if text.strip()]
    if not texts:
        raise ValueError("Нет текстов для анализа после фильтрации.")

    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words=russian_stop_words, max_features=500)
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)

    return tfidf_matrix, vectorizer


def build_average_profile(users_posts, users_subscriptions):
    """
    Создает средний профиль на основе текстов постов и подписок пользователей.

    :param users_posts: Список постов всех пользователей
    :param users_subscriptions: Список подписок всех пользователей
    :return: Средний профиль
    """
    all_texts = [text for posts in users_posts for text in posts]  # Собираем все тексты в один список
    if not all_texts:
        raise ValueError("Нет текстов для построения профиля.")

    tfidf_matrix, vectorizer = analyze_texts(all_texts)

    # Подсчитываем количество упоминаний каждой группы
    all_groups = [group for groups in users_subscriptions for group in groups]
    group_counter = Counter(all_groups)

    return {
        'tfidf_matrix': tfidf_matrix.mean(axis=0),  # Среднее значение TF-IDF
        'vectorizer': vectorizer,
        'group_stats': group_counter  # Частотный счетчик групп
    }


def compare_user_with_profile(user_posts, user_subscriptions, avg_profile):
    """
    Сравнивает данные нового пользователя с ранее построенным средним профилем.
    :param user_posts: Посты нового пользователя
    :param user_subscriptions: Подписки нового пользователя
    :param avg_profile: Средний профиль (TF-IDF и группы)
    :return: Сходство по текстам и группам
    """
    if not user_posts:
        raise ValueError("Нет текстов для анализа нового пользователя.")

    # Анализ текстов нового пользователя с использованием существующего векторизатора
    tfidf_matrix, _ = analyze_texts(user_posts, vectorizer=avg_profile['vectorizer'])
    tfidf_matrix_array = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    # Сравнение текстов на основе косинусного сходства
    text_similarity = cosine_similarity(
        [tfidf_matrix_array], 
        [np.asarray(avg_profile['tfidf_matrix']).flatten()]
    )[0][0]

    # Сравнение групп на основе количества общих групп
    group_similarity = sum(
        avg_profile['group_stats'].get(group, 0) 
        for group in user_subscriptions
    ) / max(1, len(avg_profile['group_stats']))

    return {
        'text_similarity': text_similarity,  # Сходство текстов
        'group_similarity': group_similarity  # Сходство по подпискам
    }

if __name__ == "__main__":
    user_names = ['princessss_kr', 'id175633281', 'ropik05']
    all_posts = []
    all_subscriptions = []

    for user_name in user_names:
        user_id = get_user_id(user_name)
        if user_id:
            user_posts = take_user_posts(user_id)
            user_subscriptions = get_user_subscriptions(user_id)
            all_posts.append(user_posts)
            all_subscriptions.append(user_subscriptions)

    try:
        avg_profile = build_average_profile(all_posts, all_subscriptions)
        print("Средний профиль успешно построен.")
    except ValueError as e:
        print(f"Ошибка при построении среднего профиля: {e}")
        exit()

    new_user_name = 'neadmace'
    new_user_id = get_user_id(new_user_name)
    if new_user_id:
        new_user_posts = take_user_posts(new_user_id)
        new_user_subscriptions = get_user_subscriptions(new_user_id)

        try:
            comparison = compare_user_with_profile(new_user_posts, new_user_subscriptions, avg_profile)
            print("Сравнение нового пользователя:", comparison)
        except ValueError as e:
            print(f"Ошибка анализа данных для нового пользователя: {e}")