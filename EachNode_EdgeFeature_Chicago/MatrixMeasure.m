%% Find the error in c.d.f and p.d.f
clear all; close all; clc;
%ECDF = load('TestSupT.csv');
ECDF = load(fullfile(pwd,'Results_Test','testCum3.csv'));
PCDF = load(fullfile(pwd,'Results_Test','testPredictCumSupT_train3_test3.csv'));
ECDF(1:24)
ECDF(ECDF < 0) = 0;
PCDF(PCDF < 0) = 0;
%ECDF = ECDF(end-61*48+1:end);
ECDF = ECDF(end-61*24+1:end);
PCDF = PCDF(end-61*48+1:end);

%ECDF = ECDF(2:2:end);
PCDF = PCDF(2:2:end);

PCDF = floor(PCDF + 0.5);

%PCDF = zeros(size(ECDF));
ErrCDF = ECDF - PCDF;
RMSECDF = sqrt(ErrCDF'*ErrCDF/length(ErrCDF))


T = 24; numDay = 61;
%{
for i = 1:numDay
    for j = 2:T
        if PCDF((i-1)*T+j) < PCDF((i-1)*T+j-1)
            PCDF((i-1)*T+j) = PCDF((i-1)*T+j-1);
        end
        
        if ECDF((i-1)*T+j) < ECDF((i-1)*T+j-1)
            ECDF((i-1)*T+j) = ECDF((i-1)*T+j-1);
        end
    end
end
%}

EPDF = ECDF;
PPDF = PCDF;
for i = 1:numDay
    for j = 2:T
        EPDF((i-1)*T+j) = ECDF((i-1)*T+j) - ECDF((i-1)*T+j-1);
        PPDF((i-1)*T+j) = PCDF((i-1)*T+j) - PCDF((i-1)*T+j-1);
    end
end

EPDF(EPDF < 0) = 0;
PPDF(PPDF < 0) = 0;

EPDF(1:24)
ErrPDF = EPDF - PPDF;
RMSEPDF = sqrt(ErrPDF'*ErrPDF/length(ErrPDF))
RMSEZERO = sqrt(EPDF'*EPDF/length(EPDF))


TestEPDF = EPDF;
TestPPDF = PPDF;
%% Anomaly prediction with allowed delay
AnoAcc = zeros(3, 3);
for Thres = 1:3
    Index = zeros(size(TestEPDF));
    Index(TestEPDF > Thres - 0.001) = 1;
    Index = reshape(Index, 24, numDay);
    Index1 = zeros(size(TestPPDF));
    Index1(TestPPDF >= Thres - 0.5) = 1;
    Index1 = reshape(Index1, 24, numDay);
    
    for Delay = 0:2
        IndexRes = Index1;
        for iter = 1:Delay
            Indextmp = Index1;
            Indextmp(iter+1:end, :) = Indextmp(1:end-iter, :);
            Indextmp(1:iter, :) = 0;
            IndexRes = IndexRes | Indextmp;
        end
        TotalAno = nnz(Index);
        PredAno = nnz(Index~=0 & IndexRes~=0);
        Acc = PredAno/TotalAno;
        AnoAcc(Thres, Delay+1) = Acc;
    end
end

%Plot
imagesc(AnoAcc);            %Create a colored plot of the matrix value
%colormap(flipud(gray));     %Change the colormap to gray (higher value are black and lower values are white)
colormap(colormap);

textStrings = num2str(AnoAcc(:), '%0.2f');      %Create strings from the matrix value
textStrings = strtrim(cellstr(textStrings));    %Remove any space padding
[x, y] = meshgrid(1:3);                         %Create x and y coordinates for the strings
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', 'FontSize', 20);   %Plot the strings
xlabel('Delay', 'FontSize', 20)
ylabel('Threshold', 'FontSize', 20)

set(gca,'XTick',1:4,...                         %# Change the axes tick marks
        'XTickLabel',{'0','1','2'},...  %#   and tick labels
        'YTick',1:4,...
        'YTickLabel',{'1','2','3'},...
        'TickLength',[0 0]);
set(gca, 'FontSize', 20)