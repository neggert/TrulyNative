import click
from bs4 import BeautifulSoup as bs
from zipfile import ZipFile
from urllib.parse import urlparse
from os.path import splitext, basename
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def urlparse_catch(url):
    try:
        return urlparse(url)
    except ValueError:
        return None


def parse(soup):
    result = {}

    # Title
    result['title_text'] = soup.title.text if soup.title else ''
    result['title_word_len'] = len(soup.title.text.strip()) if soup.title else 0
    result['title_char_len'] = len(soup.title.text.split()) if soup.title else 0

    # Body Text
    ' '.join([p.text for p in soup.find_all('p')])
    result['para_text'] = ' '.join([p.text for p in soup.find_all('p')])
    # result['para_word_len'] = len(soup.get_text())
    # result['para_char_len'] = len(soup.get_text().split())

    # Links
    links = [a.get('href') for a in soup.find_all('a')]
    parsed_links = [urlparse_catch(l) for l in links]
    result['link_num'] = len(links)
    result['link_resources'] = [str(p.netloc).replace('.', '_') for p in parsed_links if p]
    result['link_schemes'] = [str(p.scheme) for p in parsed_links if p]

    # Images
    images = [img.get('src') for img in soup.find_all('img')]
    parsed_images = [urlparse_catch(i) for i in images]
    result['img_num'] = len(images)
    result['img_resources'] = [str(p.netloc).replace('.', '_') for p in parsed_images if p]
    result['img_schemes'] = [str(p.scheme).replace('.', '_') for p in parsed_images if p]
    result['img_exts'] = [str(splitext(p.path)[1][1:]) for p in parsed_images if p]

    return result


def make_sklearn_features(results):
    """
    obsolete. All results don't fit in memory at once
    """
    transforms = {}

    # TF-IDF on title text
    transforms['title_tfidf'] = TfidfVectorizer()
    title_tfidf = transforms['title_tfidf'].fit_transform([r['title_text'] for r in results])

    # TF-IDF on body text
    transforms['para_tfidf'] = TfidfVectorizer()
    para_tfidf = transforms['para_tfidf'].fit_transform([r['para_text'] for r in results])

    # TF-IDF on URL resources
    transforms['url_resource_tfidf'] = TfidfVectorizer()
    url_domain_tfidf = transforms['url_resource_tfidf'].fit_transform(
        [' '.join(r['link_resources']) for r in results]
    )

    # TF-IDF on URL schemes
    transforms['url_scheme_tfidf'] = TfidfVectorizer()
    url_scheme_tfidf = transforms['url_scheme_tfidf'].fit_transform(
        [' '.join(r['link_schemes']) for r in results]
    )

    # TF-IDF on Image resources
    transforms['img_resource_tfidf'] = TfidfVectorizer()
    img_domain_tfidf = transforms['img_resource_tfidf'].fit_transform(
        [' '.join(r['img_resources']) for r in results]
    )

    # TF-IDF on img schemes
    transforms['img_scheme_tfidf'] = TfidfVectorizer()
    img_scheme_tfidf = transforms['img_scheme_tfidf'].fit_transform(
        [' '.join(r['img_schemes']) for r in results]
    )

    # TF-IDF on img extentions
    transforms['img_ext_tfidf'] = TfidfVectorizer()
    img_ext_tfidf = transforms['img_ext_tfidf'].fit_transform(
        [' '.join(r['img_exts']) for r in results]
    )

    dense_features = np.vstack((
        np.asarray([r['title_word_len'] for r in results]),
        np.asarray([r['title_char_len'] for r in results]),
        np.asarray([r['para_word_len'] for r in results]),
        np.asarray([r['para_char_len'] for r in results]),
        np.asarray([r['link_num'] for r in results]),
        np.asarray([r['img_num'] for r in results])
    )).T

    filenames = [r['filename'] for r in results]

    features = sparse.hstack((title_tfidf,
                              para_tfidf,
                              url_domain_tfidf,
                              url_scheme_tfidf,
                              img_domain_tfidf,
                              img_ext_tfidf,
                              sparse.coo_matrix(dense_features)))
    print(features.shape)


def sanitize(s):
    """Remove characters that VW won't like"""
    return s.replace(':', '<colon>').replace('|', '<pipe>').replace('\n', ' ')


def make_vw_features(request):
    tag = '\'' + request['filename']
    title = '|title ' + sanitize(request['title_text'])
    para = '|para ' + sanitize(request['para_text'])
    link_resources = '|link_resources ' + sanitize(' '.join(request['link_resources']))
    link_schemes = '|link_schemes ' + sanitize(' '.join(request['link_schemes']))
    img_resources = '|img_resources ' + sanitize(' '.join(request['img_resources']))
    img_exts = '|img_exts ' + sanitize(' '.join(request['img_exts']))

    return ' '.join((
        tag,
        title,
        para,
        link_resources,
        link_schemes,
        img_resources,
        img_exts
    ))


def get_target(filename, targets):
    try:
        target = targets.loc[filename, '0']
        return '1 ' if target == 1 else '-1 '
    except KeyError:
        return ''


@click.command()
@click.option('-n', default=-1, help='Number of documents to process')
def main(n):
    results = []

    files = [
        'data/zip/0.zip',
        'data/zip/1.zip',
        'data/zip/2.zip',
        'data/zip/3.zip',
        'data/zip/4.zip',
    ]

    targets = pd.read_csv('data/train_no_holdout.csv', index_col=0)

    with open('intermediate/vw.txt', 'w') as of:
        for zfn in files:
            print(zfn)
            with ZipFile(zfn) as zf:
                contents = zf.namelist()
                print(len(contents))
                if n > 0:
                    contents = contents[:n]
                with click.progressbar(contents) as bar:
                    for fn in bar:
                        if basename(fn):
                            with zf.open(fn) as f:
                                soup = bs(f, 'lxml')
                                r = parse(soup)
                                r['filename'] = basename(fn)
                                target = get_target(r['filename'], targets)
                                vw_example = target + make_vw_features(r)
                                of.write(vw_example + '\n')


if __name__ == '__main__':
    main()

