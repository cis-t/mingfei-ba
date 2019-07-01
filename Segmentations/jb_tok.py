# -*- coding: utf-8 -*-
import re
import jieba


def segment_with_jieba(inp_, out_):
    with open(inp_, 'r') as file:
        # read in the unsegmented lines
        result = re.sub(r' \—\"\—|世 界 文 学 名 著 百 部|红 与 黑|\,|\—\!\—|\—\!\"\—|\—\!(.)\!\—| \—\!\"\#\—|\—\!\"\#\—| \—\"\#\$\—| \—\!\!\"\—|\—\"\#\"\—|\—\%\&\’\—|" "|\—.*\—', ' ', file.read())
        result = re.sub(r'\n\n*','', result)
        verses = re.sub(r'[?。! ; ,]','\n',result)
        segmented_verses = jieba.cut(verses, cut_all=False)

    # write out segmented results
    with open(out_, 'w') as f:
        f.write('{}\n'.format(' '.join(segmented_verses)))
    print('Segmented file written to {}'.format(out_))


if __name__ == '__main__':
    
    segment_with_jieba(
        inp_='../data/test/01.txt',
        out_='../data/result/segmented_01_jieba.txt'
    )



    segment_with_jieba(
        inp_='../data/test/02.txt',
        out_='../data/result/segmented_02_jieba.txt'
    )
