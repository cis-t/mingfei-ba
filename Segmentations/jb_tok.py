# -*- coding: utf-8 -*-
import re
import jieba


def segment_with_jieba(inp_, out_):
    with open(inp_, 'r') as file:
        # read in the unsegmented lines

        result = re.sub(r' \—\"\—|世 界 文 学 名 著 百 部|红 与 黑|\—\!\—|\—\!\"\—|\—\!(.)\!\—| \—\!\"\#\—|  \—\!\"\#\—| \—\"\#\$\—| \—\!\!\"\—|\—\"\#\"\—|\—\%\&\’\—|\—.*\—', ' ', file.read())
        result = re.sub(r'\n\n\n*', '', result)
        verses = re.split(r'[?。!; ]', result)
        segmented_verses = jieba.cut(' '.join(verses), cut_all=False)
    
    # write out segmented results
    with open(out_, 'w') as f:
        for l in segmented_verses:
            f.write(re.sub(r' ', '', '{}\n'.format(' '.join(l))))
    print('Segmented file written to {}'.format(out_))




if __name__ == '__main__':
    segment_with_jieba(
        inp_='../data/test/01.txt',
        out_='../data/result/segmented_01_jieba.txt'
    )
