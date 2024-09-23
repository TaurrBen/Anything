#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>

#define MAX_SAMPLE_WEB_NUM 1000 //待测试数据网页数
#define HAMMING_THRESHOLD 3 //汉明距离阈值  
#define ARTICLE "article.txt"
#define SAMPLE "sample.txt"
#define STOPWORD "stopwords.txt"
#define HASHVALUE "hashvalue.txt"
#define RESULT "result.txt"

typedef struct WordData{
	int count;
	char text[80];
}WordData;

typedef struct Trie {
	int isStop;
	int isleaf;//是否构成完整单词
	int isFeature;//是否为特征向量 是的话赋值为特征向量索引 
	struct WordData wordData;//数据域
	struct Trie* nextNode[26];
}Trie;

typedef struct LinkList{
	int isStop;
	struct WordData wordData;//数据域
	struct LinkList* nextNode;
}LinkList;	

int N, M;
int hashValue[10000][128];
char articleWebIdentifier[16000][50];
int articleWebNum = 0;
char sampleWebIdentifier[16000][50];
int sampleWebNum = 0;
int articleFingerprint[16000][128];
int sampleFingerprint[16000][128];
WordData articleWordList[106000];
int articleWordIndex = 0;
int hammingDistance[MAX_SAMPLE_WEB_NUM][16000];

Trie* CreatTrieNode() {
	Trie* p;
	p = (Trie*)malloc(sizeof(Trie));
	p->isStop = 0;
	p->isleaf = 0;
	p->isFeature = -1;
	p->wordData.count = 0;
	memset(p->wordData.text, '\0', sizeof(char) * 80);
	for (int i = 0; i < 26; i++)
		p->nextNode[i] = NULL;
	return p;
}//建立一个结点

//将 text存入字典树 headNode中，并标记为停用词isStop 
void AddStopWordNode(Trie* headNode, char* text) {
	for (int i = 0; text[i] != '\0'; i++) {
		int index = text[i] - 'a';
		if (headNode->nextNode[index] == NULL) {
			headNode->nextNode[index] = CreatTrieNode();
		}
		headNode = headNode->nextNode[index];
	}
	headNode->isStop = 1;
}

//将 text存入字典树 headNode中，并计入词频 
void AddWordNode(Trie* headNode, char* text) {
	Trie* p= headNode;
	for (int i = 0; text[i] != '\0'; i++) {
		int index = text[i] - 'a';
		if (p->nextNode[index] == NULL) {
			p->nextNode[index] = CreatTrieNode();
		}
		p = p->nextNode[index];
	}
	if (p->isStop == 0) {
		p->isleaf = 1;
		p->wordData.count++;
		strcpy(p->wordData.text, text);
	}
}

//将 text存入字典树 headNode中，并计入词频 
void MarkFeature(Trie* headNode, char* text,int index) {
	Trie* p= headNode;
	for (int i = 0; text[i] != '\0'; i++) {
		p = p->nextNode[text[i] - 'a'];
	}
	p->isFeature = index;
}

int FeatureIndex(Trie* headNode,char* text){
	Trie* p= headNode;
	for (int i = 0; text[i] != '\0'; i++) {
		p = p->nextNode[text[i] - 'a'];
	}
	return p->isFeature;
}

//将字典树转化为词频表 
void Trie2WordList(Trie* headNode) {
	Trie* p = headNode;
	if (p == NULL) {
		return;
	}else if(p->isleaf == 1) {
		articleWordList[articleWordIndex].count = p->wordData.count;
		strcpy(articleWordList[articleWordIndex].text, p->wordData.text);
		articleWordIndex++;
	}
	for (int i = 0; i < 26; i++) {
		Trie2WordList(p->nextNode[i]);
	}
}

//释放字典树空间 
void FreeTrie(Trie* headNode) {
	for (int i = 0; i < 26; i++) {
		if (headNode->nextNode[i] != NULL)
			FreeTrie(headNode->nextNode[i]);
	}
	free(headNode);
}

//qsort的比较排序规则 
int CMP(const void* p1, const void* p2) {
	struct WordData* w1 = (struct WordData*)p1;
	struct WordData* w2 = (struct WordData*)p2;
	if (w1->count != w2->count) {//第一规则频率高
		return w2->count - w1->count;
	}
	else//第二规则字母序 
	{
		return strcmp(w1->text, w2->text);
	}
}

int isFeature(char w[]) {
	for (int i = 0; i < N; i++) {
		if (strcmp(w, articleWordList[i].text) == 0)
			return i;
	}
	return -1;
}

//获取hashvalue转0 1为-1 1 
void GetHashValue(char* filename,int maxRow,int maxCol)
{
	FILE* fp = fopen(filename, "r");
	char str[129];
	for (int row = 0; row < maxRow; row++) {
		fscanf(fp, "%s", str);
		for (int col = 0; col < maxCol; col++) {
			hashValue[row][col] = (str[col] == '1') ? 1 : -1;
		}
	}
	fclose(fp);
}

int Hamming(int* fingerprint1, int* fingerprint2, int M)
{
	int count = 0;
	for (int j = 0; j < M; j++)
	{
		int value1 = fingerprint1[j];
		int value2 = fingerprint2[j];
		//printf("%d\t%d\n", value1, value2);
		if (value1 != value2)
			count++;
		if (count > HAMMING_THRESHOLD)
			break;
	}
	//printf("Hamming distance is :%d.\n", count);
	return count;
}

int main(int argc, char** argv)
{
	char str[80];//读取流 
	N = atoi(argv[1]);
	M = atoi(argv[2]);
	//预设字典树根节点 
	Trie* trie = CreatTrieNode();
	FILE* stopWord = fopen(STOPWORD, "r");
	//获取hashvalue
	GetHashValue(HASHVALUE, N, M);
	//获取stopword
	while (!feof(stopWord)) {
		fscanf(stopWord, "%s", str);
		AddStopWordNode(trie, str);
	}
	fclose(stopWord);
	//获取全文词频表
	FILE* article = fopen(ARTICLE, "r");
	while (1)
	{
		int wordIndex = 0;
		char ch;
		while (!isalpha(ch = fgetc(article)))
		{
			if (ch == EOF)
				break;
			else 
				continue;
		}
		if (ch == EOF)
			break;
		do {
			str[wordIndex++] = tolower(ch);
		} while (isalpha(ch = fgetc(article)));
		str[wordIndex] = '\0';
		AddWordNode(trie, str);
	}
	fclose(article);
	//将树结构存为词频表 
	Trie2WordList(trie);
	//词频表排序 
	qsort(articleWordList, articleWordIndex, sizeof(struct WordData), CMP);
	//标记特征向量，并赋上索引号 
	for (int i = 0; i < N; i++) {
		MarkFeature(trie, articleWordList[i].text,i);
	}
	//获取article的指纹
	article = fopen(ARTICLE, "r");
	int articleEnd = 0;
	while (1)
	{
		fscanf(article, "%s", articleWebIdentifier[articleWebNum]);
		char ch;
		int webEnd = 0;
		while (1)
		{
			int wordIndex = 0;
			while (!isalpha(ch = fgetc(article)))
			{
				if (ch == '\f')
				{
					webEnd = 1;
					break;
				}
				else if (ch == EOF)
				{
					articleEnd = 1;
					break;
				}
				else
					continue;
			}
			if (webEnd == 1 || articleEnd == 1)
				break;
			do {
				str[wordIndex++] = tolower(ch);
			} while (isalpha(ch = fgetc(article)));
			str[wordIndex] = '\0';
			//计算网页指纹
			int index;
			//该词出现即加入对应指纹
			if((index = FeatureIndex(trie,str))>=0)
			{
				for (int col = 0; col < M; col++)
				{
					articleFingerprint[articleWebNum][col] += hashValue[index][col];
				}
			}
		}
		//将网页指纹转为01字符串
		for (int col = 0; col < M; col++)
		{
			articleFingerprint[articleWebNum][col] = (articleFingerprint[articleWebNum][col] > 0) ? 1 : 0;
		}
		articleWebNum++;
		if (articleEnd == 1)
			break;
	}
	fclose(article);
	//获取sample的指纹
	FILE* sample = fopen(SAMPLE, "r");
	int sampleArticleEnd = 0;
	while (1)
	{
		fscanf(sample, "%s", sampleWebIdentifier[sampleWebNum]);
		char ch;
		int webEnd = 0;
		while (1)
		{
			int wordIndex = 0;
			while (!isalpha(ch = fgetc(sample)))
			{
				if (ch == '\f')
				{
					webEnd = 1;
					break;
				}
				else if (ch == EOF)
				{
					sampleArticleEnd = 1;
					break;
				}
				else
					continue;
			}
			if (webEnd == 1 || sampleArticleEnd == 1)
				break;
			do {
				str[wordIndex++] = tolower(ch);
			} while (isalpha(ch = fgetc(sample)));
			str[wordIndex] = '\0';
			//该词出现即加入对应指纹
			int index; 
			if((index = FeatureIndex(trie,str))>=0)
			{
				for (int col = 0; col < M; col++)
				{
					sampleFingerprint[sampleWebNum][col] += hashValue[index][col];
				}
			}
		}
		//将网页指纹转为01字符串
		for (int col = 0; col < M; col++)
		{
			sampleFingerprint[sampleWebNum][col] = (sampleFingerprint[sampleWebNum][col] > 0) ? 1 : 0;
		}
		sampleWebNum++;
		if (sampleArticleEnd == 1)
			break;
	}
	fclose(sample);
	//计算hamming距离
	FILE* result = fopen(RESULT, "w");
	for (int sampleIndex = 0; sampleIndex < sampleWebNum - 1; sampleIndex++)
	{
		int distanceIndex[HAMMING_THRESHOLD + 1][16000];
		int index[HAMMING_THRESHOLD + 1];
		memset(distanceIndex, 0, (HAMMING_THRESHOLD + 1) * 16000 * sizeof(int));
		memset(index, 0, (HAMMING_THRESHOLD + 1) * sizeof(int));
		for (int articleIndex = 0; articleIndex < articleWebNum; articleIndex++)
		{
			int distance = Hamming(sampleFingerprint[sampleIndex],articleFingerprint[articleIndex], M);
			if (distance <= HAMMING_THRESHOLD)
			{
				distanceIndex[distance][index[distance]++] = articleIndex;
			}
		}
		//打印
		if(sampleIndex==0) 
			printf("%s\n", sampleWebIdentifier[sampleIndex]);
		fprintf(result, "%s\n", sampleWebIdentifier[sampleIndex]);
		for (int k = 0; k <= HAMMING_THRESHOLD; k++)
		{
			int l;
			if (index[k])
			{
				if(sampleIndex==0)
					printf("%d:", k);
				fprintf(result, "%d:", k);
				for (l = 0; l < index[k]; l++)
				{
					if(sampleIndex==0)
						printf("%s ", articleWebIdentifier[distanceIndex[k][l]]);
					fprintf(result, "%s ", articleWebIdentifier[distanceIndex[k][l]]);
				}
				if(sampleIndex==0)
					printf("\n");
				fprintf(result, "\n");
			}
		}
	}
	fclose(result);
}
