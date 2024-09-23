%% Author
% Author : Federico Giai Pron (federico.giaipron@gmail.com)
% Mail   : federico.giaipron@gmail.com
%% Genetic algorithm (GA)
function [Xpbest, ObjFunpbest,ObjFuncbestPlot,Scorepbest, Data] = Optimization_GA_v01(XLim, Sett, Data)
% Initialization %每个个体有5条染色体，一条染色体有318个决策变量，
X      = zeros(Sett.NumChr,Sett.LengthX);%每一行为染色体，
Xcbest = zeros(1,Sett.LengthX);%个体最优，对比每个染色体
Xpbest = zeros(1,Sett.LengthX);%种群最优，对比每个个体
ObjFun = zeros(Sett.NumChr,1);
ObjFuncbestPlot = zeros(Sett.NumPop,Sett.NumIter);
switch Sett.Type
    case 'min'
        ObjFunpbest     = +inf;
        ObjFunpbestPlot = +inf*ones(1,Sett.NumIter);
    case 'max'
        ObjFunpbest     = -inf;
        ObjFunpbestPlot = -inf*ones(1,Sett.NumIter);
end
% Calculation
for IndexPop = 1:1:Sett.NumPop
    fprintf('- IndexPop:    %3i out of %3i\n',IndexPop,Sett.NumPop);
    % X initialization初始化一个个体
    for IndexChr = 1:1:Sett.NumChr
        for IndexX = 1:1:Sett.LengthX
            X(IndexChr,IndexX) = floor(XLim(1,IndexX) + (XLim(2,IndexX) - XLim(1,IndexX))*rand(1));
        end
    end
    % Iteration进行迭代
    for IndexIter = 1:1:Sett.NumIter
        fprintf('-- IndexIter:  %3i out of %3i\n',IndexIter,Sett.NumIter);
        % ObjFun calculation
        for IndexChr = 1:1:Sett.NumChr
            fprintf('--- IndexChr:  %3i out of %3i\n',IndexChr,Sett.NumChr);
            [Score(IndexChr,:) ObjFun(IndexChr)] = ObjFun_fun(X(IndexChr,:),Data);
        end
        % 选择算子
        % Natural selection
        switch Sett.Type
            case 'min'
                [~,Indexc]  = sort(ObjFun,'ascend');
            case 'max'
                [~,Indexc]  = sort(ObjFun,'descend');
        end
        % 随机选择一个作为个体最优
        Scorecbest = Score(Indexc(1),:);
        ObjFuncbest = ObjFun(Indexc(1));%记录个体最优决策值
        Xcbest      = X(Indexc(1),:);%记录个体最优决策变量
        % 输出个体最优决策值与最优决策变量
        % Xcbest and ObjFuncbest output
        if(Sett.Fprintf==true)
            for IndexS = 1:4
                fprintf('-- Scorecbest(%d): %-+1.10e,',IndexS,Scorecbest(IndexS));
            end
            fprintf('-- Scorecbest(5): %-+1.10e,',Scorecbest(5));
            fprintf('-- ObjFuncbest: %-+1.10e,',ObjFuncbest);
            for IndexX = 1:1:Sett.LengthX
                if(IndexX < Sett.LengthX)
                    fprintf(' Xcbest(%i) = %-+1.10d,',IndexX,Xcbest(IndexX));
                else
                    fprintf(' Xcbest(%i) = %-+1.10d\n',IndexX,Xcbest(IndexX));
                end
            end
        end
        % 均匀杂交
        % Uniform crossover
        for IndexChr = 1:1:Sett.NumChr
            for IndexX = 1:1:Sett.LengthX
                Coin = round(rand(1));
                IndexParent = randi(2);
                if(IndexChr ~= Indexc(1) && IndexChr ~= Indexc(2) && Coin == true)
                    X(IndexChr,IndexX) = X(Indexc(IndexParent),IndexX);
                end
            end
        end
        % 变异
        % Mutation
        for IndexX = 1:1:Sett.LengthX
            Coin = round(rand(1));
            if(Coin == true)
                X(Indexc(end),IndexX) = floor(XLim(1,IndexX) + (XLim(2,IndexX) - XLim(1,IndexX))*rand(1));
            end
        end
        % Plot
        ObjFuncbestPlot(IndexPop,IndexIter) = ObjFuncbest;
        switch Sett.Type
            case 'min'
                if(ObjFuncbest < ObjFunpbestPlot(IndexIter))
                    ObjFunpbestPlot(IndexIter) = ObjFuncbest;
                end
            case 'max'
                if(ObjFuncbest > ObjFunpbestPlot(IndexIter))
                    ObjFunpbestPlot(IndexIter) = ObjFuncbest;
                end
        end
    end
    % Xpbest and ObjFunpbest determination
    switch Sett.Type
        case 'min'
            if(ObjFuncbest < ObjFunpbest)
                Xpbest      = Xcbest;
                ObjFunpbest = ObjFuncbest;
                Scorepbest = Scorecbest;
            end
        case 'max'
            if(ObjFuncbest > ObjFunpbest)
                Xpbest      = Xcbest;
                ObjFunpbest = ObjFuncbest;
                Scorepbest = Scorecbest;
            end
    end
    % Xpbest and ObjFunpbest output
    for IndexS = 1:4
        fprintf('-- Scorepbest(%d): %-+1.10e,',IndexS,Scorepbest(IndexS));
    end
    fprintf('-- Scorepbest(5): %-+1.10e,',Scorepbest(5));
    fprintf('-  ObjFunpbest: %-+1.10e,',ObjFunpbest);
    for IndexX = 1:1:Sett.LengthX
        if(IndexX < Sett.LengthX)
            fprintf(' Xpbest(%i) = %-+1.10d,',IndexX,Xpbest(IndexX));
        else
            fprintf(' Xpbest(%i) = %-+1.10d\n',IndexX,Xpbest(IndexX));
        end
    end
    toc
end
%% Plot
if(Sett.FlagPlots == true)
    figure;
    semilogy(ObjFunpbestPlot,'LineWidth',10);
    for IndexPop = 1:1:Sett.NumPop
        hold on;
        semilogy(ObjFuncbestPlot(IndexPop,:),'LineWidth',1.5);
        title('Objective function optimization');
        xlabel('NumIter');
        ylabel('ObjFun_{c,best}(Pop)');
        grid on;
    end
end
end