//60s
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#define max 500
int N, M; //N：需要几个单词，M：指纹长度（取多长的哈希值）
int f[4][max];
char stoplist[80];//储存停用词
//int tot[10001];

typedef struct trie {
	int isStop;
	int count;
	int isleaf;//是否构成完整单词
	struct trie *child[26];
	char word[80];
} trie;

struct worddata {
	int count;
	char word[80];
};
struct worddata wordlist[106000];
int wordlist_num = 0;
struct worddata article_wordlist[6500];
//struct worddata sample_wordlist[6500];

int weight[10001];

int getWord(FILE *fp, char w[]) {
	int c;
	int l = 0;
	while (!isalpha(c = fgetc(fp))) {
		if (c == '\f') {
			return -1;//-1表示一个网页结束
		} else if (c == EOF)
			return 0;//0文件结束
		else
			continue;
	}
	do {
		w[l++] = tolower(c);
	} while (isalpha(c = fgetc(fp)));
	w[l] = '\0';
	return 1;
}//w为取到的单词，返回1
char getWord_w[80];//从getWord函数中取到的单词char *w

trie *CreatTrie() {
	trie *p;
	p = (trie *)malloc(sizeof(trie));
	p->count = 0;
	p->isleaf = 0;
	p->isStop = 0;
	for (int i = 0; i < 26; i++)
		p->child[i] = NULL;
	return p;
}//建立trie树的一个结点

int isFeature(char w[]) {
	int i;
	for (i = 0; i < N; i++) {
		if (strcmp(w, wordlist[i].word) == 0)
			return i;
	}
	return -1;
}

void insert_stop(trie *obj, char word[]) {
	for (int i = 0; word[i]; i++) {
		int index = word[i] - 'a';
		if (obj->child[index] == NULL) {
			obj->child[index] = CreatTrie();
		}
		obj = obj->child[index];
	}
	obj->isStop = 1;
}

void wordTree1(trie *root, char w[]) {
	int l = 0;
	trie *p;
	for (p = root; w[l] != '\0'; l++) {
		if (p->child[w[l] - 'a'] == NULL) {
			p->child[w[l] - 'a'] = CreatTrie();
		}
		p = p->child[w[l] - 'a'];
	}
	if (p->isStop == 0) {
		p->isleaf = 1;
		p->count++;
		strcpy(p->word, w);
	}
}

void PrintTrie(trie *root) {
	if (root == NULL) {
		return;
	} else if (root->isleaf == 1) {
		wordlist[wordlist_num].count = root->count;
		strcpy(wordlist[wordlist_num].word, root->word);
		wordlist_num++;
	}
	for (int i = 0; i < 26; i++) {
		PrintTrie(root->child[i]);
	}
}

void trieFree(trie *obj) {
	for (int i = 0; i < 26; i++) {
		if (obj->child[i] != NULL)
			trieFree(obj->child[i]);
	}
	free(obj);
}

char article_name[16000][50];
int m = 0;
char sample_name[10000][50];
//char test_name[50];
int n = -1;

int cmp(const void *p1, const void *p2) {
	struct worddata *w1 = (struct worddata *)p1;
	struct worddata *w2 = (struct worddata *)p2;
	if (w1->count != w2->count) {
		return w2->count - w1->count;
	} else
		return strcmp(w1->word, w2->word);
}

char hashvalue_str[10001][130];//取到的哈希值

int article_fingerprint[16000][130];

int sample_fingerprint[10000][130];

int HanMing(int a[], int b[]) {
	int sum = 0;
	for (int k = 0; k < M; k++) {
		if (a[k] != b[k])
			sum++;
		if (sum > 3)
			break;
	}
	return sum;
}//计算汉明距离

int HanMinglen[max][16000]; //sample，article
char line[10001];

int main(int argc, char *argv[]) {
	N = atoi(argv[1]);
	M = atoi(argv[2]);
	FILE *out = fopen("result.txt", "w");
	FILE *sample, *article, *hash, *stop;
	article = fopen("article.txt", "r");
	hash = fopen("hashvalue.txt", "r");
	stop = fopen("stopwords.txt", "r");
	for (int k = 0; k < N; k++) {
		fscanf(hash, "%s", line);
		for (int s = 0; s < M; s++) {
			hashvalue_str[k][s] = (line[s] == '1') ? 1 : -1;
		}
	}
	fclose(hash);
//环节1.1：取哈希值
	trie *root = CreatTrie();
	while (fscanf(stop, "%s", stoplist) != EOF) {
		insert_stop(root, stoplist);
	}
	fclose(stop);
//环节1.2：取停用词
	int t;
	while ((t = getWord(article, getWord_w)) != 0) {
		if (t == 1) {
			wordTree1(root, getWord_w);
		}
	}
	PrintTrie(root);//构建wordlist完毕
	qsort(wordlist, wordlist_num, sizeof(struct worddata), cmp); //排序结束
	trieFree(root);
//	trie *Root = CreatTrie();
//	for (int i = 0; i < N; i++) {
//		insert_N(Root, wordlist[i].word, i);
//	}
	fclose(article);
//环节2：取整篇文件的词频统计前N个
	article = fopen("article.txt", "r");
	int c;
	int d;
	int flag1 = 0;

	while (1) { //整个文件
		fscanf(article, "%s", article_name[++m]);
		for (int i = 0; i < N; i++) {
			article_wordlist[i].count = 0;
		}
		while ((c = getWord(article, getWord_w)) != -1) {
			if (c == 0) {
				flag1 = 1;
				break;
			} else if ((d = isFeature(getWord_w)) >= 0) {
//				for (int j = 0; j < M; j++) {
//					article_fingerprint[m][j] += hashvalue_str[d][j];//每个词都加入指纹
//				}
			}
		}//一个网页内特征向量统计
		for (int i = 0; i < M; i++) {
			article_fingerprint[m][i] = (article_fingerprint[m][i] > 0) ? 1 : 0;
		}
		if (flag1 == 1)
			break;
	}
////环节3：article的指纹统计
//	sample = fopen("sample.txt", "r");
//	flag1 = 0;
//	while (1) { //整个文件
//		fscanf(sample, "%s", sample_name[++n]);//是下标为n的sample
//		int temp;
//		while ((temp = getWord(sample, getWord_w)) != -1) {
//			if (temp == 0) {
//				flag1 = 1;
//				break;
//			}
////			int feature_num;
//			if ((d = isFeature(getWord_w)) >= 0) {
//				for (int j = 0; j < M; j++) {
//					sample_fingerprint[n][j] += hashvalue_str[d][j];//每个词都加入指纹
//				}
//			}
//		}
//		if (flag1 == 1)
//			break;
//		for (int i = 0; i < M; i++) {
//			sample_fingerprint[n][i] = (sample_fingerprint[n][i] > 0) ? 1 : 0;
//		}
//	}
////环节4：复制一份sample的
//	memset(HanMinglen, -1, sizeof(HanMinglen));
//	for (int k = 0; k < n; k++) { //sample
//		for (int l = 0; l < m ; l++) {//article
//			if (HanMing(sample_fingerprint[k], article_fingerprint[l]) == 0) {
//				HanMinglen[k][l] = 0;
//				f[0][k] = 1;
//			} else if (HanMing(sample_fingerprint[k], article_fingerprint[l]) == 1) {
//				HanMinglen[k][l] = 1;
//				f[1][k] = 1;
//			} else if (HanMing(sample_fingerprint[k], article_fingerprint[l]) == 2) {
//				HanMinglen[k][l] = 2;
//				f[2][k] = 1;
//			} else if (HanMing(sample_fingerprint[k], article_fingerprint[l]) == 3) {
//				HanMinglen[k][l] = 3;
//				f[3][k] = 1;
//			}
//		}
//	}//计算汉明距离
//
//	for (int k = 0; k < 1; k++) { //sample
//		printf("%s\n", sample_name[k]);
//		for (int z = 0; z <= 3; z++) {//有没有这一行
//			if (f[z][k] == 1) {
//				printf("%d:", z);
//				for (int l = 0; l < m; l++) { //article
//					if (HanMinglen[k][l] == z) {
//						printf("%s ", article_name[l]);
//					}
//				}
//				printf("\n");
//			}
//		}
//	}
//	for (int k = 0; k < n; k++) { //sample
//		fprintf(out, "%s\n", sample_name[k]);
//		for (int z = 0; z <= 3; z++) {//有没有这一行
//			if (f[z][k] == 1) {
//				fprintf(out, "%d:", z);
//				for (int l = 0; l < m; l++) { //article
//					if (HanMinglen[k][l] == z) {
//						fprintf(out, "%s ", article_name[l]);
//					}
//				}
//				fprintf(out, "\n");
//			}
//		}
//	}
//
////环节5：汉明距离
//	return 0;
}
