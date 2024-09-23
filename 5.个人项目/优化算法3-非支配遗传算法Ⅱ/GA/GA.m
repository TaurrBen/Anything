clc
clear
close all
b=[3 2 2 2 3 3 3 3 3 3 1 1 3 1 3 2 1];
KT=[7 5	12	4	34	10	8	10	6	17	6	1	25	420	10	10	20;5 4	10	3 32 9 7	8	5	16	0	0	23	0	8	8	0;
3 0	0	0 30 8 6 6 4 15	0 0	21	0 6	0 0;];  
KC=[11.67 8.33 20.00 6.67 437.18 16.67 13.34	32.62	10.00	247.43	10.18	32.28	41.67	420.00	16.67	16.83	21.03;
11.67 9.34 30.01 7.00 435.12	21.00	16.34	31.95	11.67	254.67	0	0	53.68	0	18.67 24.161226 0;
9.00 0 0 0 433.80	24.01	18.00	28.62	12.00	265.69	0	0	63.02	0	18.00	0 0;];
KE=[0.0432	0.0308	0.0740	0.0247	361.0063	0.0617	0.0493	0.4799	0.0370	207.1352	3.9012	13.6991	0.1542	0.0000	0.0617	3.4965	22.8987;
0.0617	0.0493	0.1850	0.0370	361.0063	0.1110	0.0863	0.4923	0.0617	207.2215	0	0	0.2837	0	0.0987	3.5828	0;
0.0555	0	0	0 361.0515	0.1480	0.1110	0.4800	0.0740	210.5110	0	0	0.3885	0	0.1110	0	0;];
% b=[3 2 1 3 2 2 3 3 3 2 3 2 1 3 2 3];
% KT=[6 5	12 4 35 10 8 10 7 15 6 1 25 420 10 12;6 5 12 4 30 10 6 5 5 15 6 1 20 420 7 10;4 4 8 2 25 8 4 4 3 12 6 1 15 420 5 8];  
% KC=[6.50 5.42 13.00	4.33 396.28	10.83 8.67 15.11 7.58 221.54 6.55 31.71	27.09 420.00 10.83 13.25;6.50 5.42 13.00 4.33 408.74 10.83 7.00 9.70 5.83 225.55 6.55 31.71 23.33 420.00 9.00 11.04;4.67 4.67 10.00 2.33 403.83 9.33 5.00 8.61 3.75 226.40 6.55 31.71 18.75 420.00 6.25 10.17];
% KE=[0.0058 0.0048 0.0115 0.0038	348.3473 0.0096	0.0077 0.3237 0.0067 206.9181 1.0046 13.8047 0.0240	0 0.0096 4.6216; 0.0058	0.0048 0.0115 0.0038 363.5984 0.0096 0.0115	0.3189 0.0038 210.1336 1.0046 13.8047 0.0383 0 0.0230 4.6292;0.00767 0.00767 0.02300 0.00382 363.59049	0.01532	0.01150	0.31795	0.00865	204.132 1.00455	13.80466 0.04313 0 0.01440 4.63300];
% n=21;
% b=[3 2 1 3 2 2 3 3 3 2 3 2 1 3 2 3 2 2 3 3 2];
% KT=[25 7 5 12 7 7 25 25 20 10 20 10 5 20 8 12 420 15 12 15 10;20 5 0 10 5 5 20 20 15 5 15 8 0 15 5 10 360 10 10 10 8;17 0 0 8 0 0 15 18 13 0 12 0 0 10 0 8 0 0 7 8 0];  
% KC=[97 19 13 32 19 15 583 152 122 27 468 443 13 37 15 22 420 40 22 40 27; 105 22 0 44 22 15 604 155 117 22 471 460 0 40 18 27 360 44 27 44 28; 111 0 0 35 0 0 607 170 122 0 473 0 0 44 0 35 0 0 31 49 0]; 
% KE=[3.10 0.02 0.25 0.07 0.02 0.06 307.84 0.76 0.18 0.03 0.12 452.27 0.61 6.72 0.01 2.41 0 10.4 0.2 7.64 11.68; 2.51 0.03 0 0.09 0.03 0.05 307.88 0.78 0.18 0.03 0.14 460.14 0 5.04 0.02 2.04 5.64 6.96 0.34 5.12 9.36; 2.16 0 0 0.07 0 0 307.90 0.81 0.20 0 0.11 0 0 3.37 0 1.64 0 0 0.47 4.12 0]; 
n=length(b);B=sum(b);cum_b=cumsum(b);
NumPop=1;%群体数
NumChr=30;%染色体数
NumIter=500;%迭代次数
LengthX=n;%决策变量长度
X=zeros(NumChr,LengthX);%染色体群体X

Zt=zeros(NumIter,NumChr,NumPop); Zc=zeros(NumIter,NumChr,NumPop); Ze=zeros(NumIter,NumChr,NumPop); aim=zeros(NumIter,NumChr,NumPop);Fitness=zeros(1,NumChr);NewFitness=zeros(1,NumChr);
Ztx=zeros(1,NumPop);Ztn=zeros(1,NumPop);Zcx=zeros(1,NumPop);Zcn=zeros(1,NumPop);Zex=zeros(1,NumPop);Zen=zeros(1,NumPop);
Tabu=zeros(NumChr,LengthX,NumIter);                        %记录每一次循环中各蚂蚁选择的方案 
Pareto1=zeros(NumIter,4,NumPop);
aimbest=zeros(NumIter,NumPop);                                    %每次循环的最优综合值 
Ztbest=zeros(NumIter,NumPop);                             %每次循环最优综合值对应的时间 
Zcbest=zeros(NumIter,NumPop);                             %每次循环最优综合值对应的成本 
Zebest=zeros(NumIter,NumPop);                           %每次循环最优综合值对应的碳排放
Tabubest=zeros(NumIter,LengthX,1);
z=zeros(NumIter,NumPop);                                     %每次循环最优的染色体的编号 
Ztaverage=zeros(NumIter,NumPop);                          %计算每次迭代所有染色体时间均值 
Zcaverage=zeros(NumIter,NumPop);                          %计算每次迭代所有染色体成本均值
Zeaverage=zeros(NumIter,NumPop);                        %计算每次迭代所有染色体碳排放均值

for IndexPop = 1:1:NumPop
    tic
    fprintf('- IndexPop:    %3i out of %3i\n',IndexPop,NumPop);

    povertime = zeros(1, LengthX); % 预分配 povertime 向量
    % X 初始化染色体群体，有NumChr条染色体，一条染色体为一个解。并调用函数计算时间、成本和碳排放
    for IndexChr = 1:1:NumChr
        X(IndexChr,:) = ceil(rand(1, LengthX) .* b);
    end

    % 进行迭代
    for IndexIter = 1:1:NumIter
        fprintf('-- IndexIter:  %3i out of %3i\n',IndexIter,NumIter);
    
        %1.适应度计算
        %计算时间、成本和碳排放 
        for IndexChr = 1:1:NumChr
            for IndexX = 1:1:LengthX
                povertime(IndexX) = KT(X(IndexChr,IndexX),IndexX);
                povertime1(IndexX) = KC(X(IndexChr,IndexX),IndexX);
                povertime2(IndexX) = KE(X(IndexChr,IndexX),IndexX);
                if IndexX == 1 % 将选择的方案记录在 Tabu 矩阵中
                    Tabu(IndexChr, IndexX, IndexIter) = X(IndexChr,1);
                else
                    Tabu(IndexChr, IndexX, IndexIter) = sum(b(1:IndexX-1)) + X(IndexChr,IndexX);
                end
            end
            Zt(IndexIter,IndexChr,IndexPop) = time(povertime);
            Zc(IndexIter,IndexChr,IndexPop) = cost(KC, n, X(IndexChr,:));
            Ze(IndexIter,IndexChr,IndexPop) = round(emission(KE, n, X(IndexChr,:)));
        end
    
        Ztx(IndexIter,IndexPop)=max(Zt(IndexIter,:,IndexPop));
        Ztn(IndexIter,IndexPop)=min(Zt(IndexIter,:,IndexPop));
        Zcx(IndexIter,IndexPop)=max(Zc(IndexIter,:,IndexPop));
        Zcn(IndexIter,IndexPop)=min(Zc(IndexIter,:,IndexPop));
        Zex(IndexIter,IndexPop)=max(Ze(IndexIter,:,IndexPop));
        Zen(IndexIter,IndexPop)=min(Ze(IndexIter,:,IndexPop));
        %%计算随机动态权重 
        r1=rand; 
        r2=rand;
        r3=rand;
        r=r1+r2+r3;
        wt=r1/r;
        wc=r2/r;
        we=max(max(r1,r2),r3)/r; 
        %%计算目标综合值 
        rand_spe=rand(1,NumChr);
        %取为-aim作为适应度越大越好
        aim(IndexIter,:,IndexPop)=wc*(Zc(IndexIter,:,IndexPop)-Zcn(IndexIter,IndexPop)+rand_spe)./(Zcx(IndexIter,IndexPop)-Zcn(IndexIter,IndexPop)+rand_spe)+wt*(Zt(IndexIter,:,IndexPop)-Ztn(IndexIter,IndexPop)+rand_spe)./(Ztx(IndexIter,IndexPop)-Ztn(IndexIter,IndexPop)+rand_spe)+we*(Ze(IndexIter,:,IndexPop)-Zen(IndexIter,IndexPop)+rand_spe)./(Zex(IndexIter,IndexPop)-Zen(IndexIter,IndexPop)+rand_spe); 
        Fitness=1./aim(IndexIter,:,IndexPop);
        aimbest(IndexIter,IndexPop)=aim(IndexIter,1,IndexPop);                                       %假设第一条染色体为最优 
        Ztbest(IndexIter,IndexPop)=Zt(IndexIter,1,IndexPop);                                          %记录第一条染色体的时间 
        Zcbest(IndexIter,IndexPop)=Zc(IndexIter,1,IndexPop);                                         %记录第一条染色体的成本 
        Zebest(IndexIter,IndexPop)=Ze(IndexIter,1,IndexPop);                                       %记录第一条染色体的碳排放  
        Tabubest(IndexIter,:,1)=Tabu(1,:,IndexIter);
        z(IndexIter,IndexPop)=1;                                                    %记录第一条染色体的序号
        
        
        for IndexChr=2:NumChr
            if aimbest(IndexIter,IndexPop)>aim(IndexIter,IndexChr,IndexPop) 
               aimbest(IndexIter,IndexPop)=aim(IndexIter,IndexChr,IndexPop); 
               Zcbest(IndexIter,IndexPop)=Zc(IndexIter,IndexChr,IndexPop); 
               Zebest(IndexIter,IndexPop)=Ze(IndexIter,IndexChr,IndexPop); 
               Ztbest(IndexIter,IndexPop)=Zt(IndexIter,IndexChr,IndexPop); 
               Tabubest(IndexIter,:,1)=Tabu(IndexChr,:,IndexIter); 
               z(IndexIter,IndexPop)=IndexChr;
            else
            end
        end 
        Pareto1(IndexIter,1,IndexPop)=aimbest(IndexIter,IndexPop);
        Pareto1(IndexIter,2,IndexPop)=Ztbest(IndexIter,IndexPop); 
        Pareto1(IndexIter,3,IndexPop)=Zcbest(IndexIter,IndexPop); 
        Pareto1(IndexIter,4,IndexPop)=Zebest(IndexIter,IndexPop); 
        Ztaverage(IndexIter,IndexPop)=sum(Zt(IndexIter,:,IndexPop))/NumChr; 
        Zcaverage(IndexIter,IndexPop)=sum(Zc(IndexIter,:,IndexPop))/NumChr; 
        Zeaverage(IndexIter,IndexPop)=sum(Ze(IndexIter,:,IndexPop))/NumChr;

        %2.自然选择
        %根据适应度做轮盘赌
        P = Fitness./sum(Fitness);
        PCum = cumsum(P);
        NewX=zeros(NumChr,LengthX);
        for IndexChr = 1:1:NumChr
            r1 = rand(1);
            choice = sum(r1>PCum)+1;
            NewX(IndexChr,:)=X(choice,:);
            NewFitness(IndexChr)=Fitness(choice);
        end
        %选择出新子代
        X=NewX;
        Fitness=NewFitness;
        [~,Indexc]  = sort(Fitness,'descend');%Indexc为排序后的索引表

        %3.杂交算子
        
        for IndexChr = 1:1:NumChr
            x1 = zeros(1,LengthX);x2 = zeros(1,LengthX);
%             % 3.1选择性与最优秀的染色体进行杂交
%             for IndexX = 1:1:LengthX
%                 Coin = round(rand(1));
%                 IndexParent = randi(LengthX);
%                 if(IndexChr ~= Indexc(1) && IndexChr ~= Indexc(2) && Coin == true)
%                     X(IndexChr,IndexX) = X(Indexc(IndexParent),IndexX);
%                 end
%             end
            % 3.2选择性与某个染色体进行片段杂交
            Coin = rand(1);
            if(Coin > 0.3)
                Index1=randi(LengthX);
                Index2=randi(LengthX);
                IndexParent=randi(LengthX);
                X(IndexChr,Index1:Index2)=X(Indexc(IndexParent),Index1:Index2);
            end
%             % 3.3选择性与某个染色体进行片段杂交
%             Coin = rand(1);
%             if(Coin > 0.3)
%                 Index1=randi(LengthX);
%                 Index2=randi(LengthX);
%                 IndexParent1=randi(LengthX);
%                 IndexParent2=randi(LengthX);
%                 x1=X(Indexc(IndexParent1),:);
%                 x2=X(Indexc(IndexParent2),:);
%                 NewX(IndexChr,:) = x1;
%                 x1(Index1:Index2)=x2(Index1:Index2);
%                 x2(Index1:Index2)=NewX(IndexChr,Index1:Index2);
%                 if(randi(1))
%                     NewX(IndexChr,:) = x1;
%                 else
%                     NewX(IndexChr,:) = x2;
%                 end
%             end
        end
        X=NewX;%配合3.3更新
        %4.变异算子
        % 4.1选择性的将最差的染色体进行变异
        for IndexX = 1:1:LengthX
            Coin = rand(1);
            if(Coin > 0.9)
                X(Indexc(end),IndexX) = ceil(rand * b(IndexX));
            end
        end
%         % 4.2选择性的将染色体进行变异
%         for IndexChr = 1:1:NumChr
%             for IndexX = 1:1:LengthX
%                 Coin = rand(1);
%                 if(Coin > 0.9)
%                     X(IndexChr,IndexX) = ceil(rand * b(IndexX));
%                 end
%             end
%         end
    end
    toc
end

%%  建立Pareto解集
% 取第一代
IndexPop=1;
Pareto=inf.*ones(NumIter,3);
tabupareto=zeros(NumIter,LengthX,1);
Pareto(1,1)=Zt(1,1,IndexPop); 
Pareto(1,2)=Zc(1,1,IndexPop); 
Pareto(1,3)=Ze(1,1,IndexPop);
tabupareto(1,:,1)=Tabu(1,:,1);
k=1; 
for l=1:NumChr
    for i=1:NumIter
        for j=1:NumIter  
            if Zt(i,l,IndexPop)<Pareto(j,1)||Zc(i,l,IndexPop)<Pareto(j,2)||Ze(i,l,IndexPop)<Pareto(j,3)  
            else 
                break 
            end 
        end 
        if j==NumIter 
           k=k+1;  
           Pareto(k,1)=Zt(i,l,IndexPop); 
           Pareto(k,2)=Zc(i,l,IndexPop);  
           Pareto(k,3)=Ze(i,l,IndexPop);
           t=NumIter*(l-1)+i;
           temp1=ceil(t/NumIter);  
           temp2=t-(ceil(t/NumIter)-1)*NumIter;  
           tabupareto(k,:,1)=Tabu(temp1,:,temp2);
           ttt = Tabu(l,:,i);
        end  
    end
end
x=zeros(NumIter);
for i=1:NumIter 
    x(i)=i; 
end 


%%设置阈值，筛选Pareto解 
Paretobest=inf.*ones(NumIter,3); 
tabuparetobest=zeros(NumIter,LengthX,1); 
tmax=550;cmax=2600;emax=800; 
temp3=0; 
for i=1:k
    if((Pareto(i,1)<=tmax)&&(Pareto(i,2)<=cmax)&&(Pareto(i,3)<=emax)) 
        temp3=temp3+1;
        Paretobest(temp3,1)=Pareto(i,1);
        Paretobest(temp3,2)=Pareto(i,2); 
        Paretobest(temp3,3)=Pareto(i,3); 
        tabuparetobest(temp3,:,1)=tabupareto(i,:,1);
    else
    end
end  


%%作图：迭代次数与总时间平均值、总成本平均值和总碳排放平均值的关系 
figure(1); 
plot(x,Ztaverage,'k');
xlabel('迭代次数'); 
ylabel('总时间平均值'); 
title('总时间平均值与迭代次数的关系'); 

figure(2); 
plot(x,Zcaverage,'k'); 
xlabel('迭代次数'); 
ylabel('总成本平均值');
title('总成本平均值与迭代次数的关系');  

figure(3); plot(x,Zeaverage,'k'); 
xlabel('迭代次数'); 
ylabel('总碳排放平均值'); 
title('总碳排放平均值与迭代次数的关系'); 
