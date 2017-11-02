TrainType3_TestType1 = load(fullfile(pwd,'Results_Matrix_Aver','AnoAcc_TrainType2_TestType1.csv'));
TrainType3_TestType2 = load(fullfile(pwd,'Results_Matrix_Aver','AnoAcc_TrainType2_TestType2.csv'));
TrainType3_TestType3 = load(fullfile(pwd,'Results_Matrix_Aver','AnoAcc_TrainType2_TestType3.csv'));

subplot(1,3,1)
imagesc(TrainType3_TestType1);            %Create a colored plot of the matrix value
%colormap(flipud(gray));     %Change the colormap to gray (higher value are black and lower values are white)
colormap(colormap);

textStrings = num2str(TrainType3_TestType1(:), '%0.2f');      %Create strings from the matrix value
textStrings = strtrim(cellstr(textStrings));    %Remove any space padding
[x, y] = meshgrid(1:3);                         %Create x and y coordinates for the strings
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', 'FontSize', 20);   %Plot the strings
xlabel('Delay(Type1)', 'FontSize', 20)
ylabel('Threshold', 'FontSize', 20)

set(gca,'XTick',1:4,...                         %# Change the axes tick marks
        'XTickLabel',{'0','1','2'},...  %#   and tick labels
        'YTick',1:4,...
        'YTickLabel',{'1','2','3'},...
        'TickLength',[0 0]);
set(gca, 'FontSize', 20)

subplot(1,3,2)
imagesc(TrainType3_TestType2);            %Create a colored plot of the matrix value
%colormap(flipud(gray));     %Change the colormap to gray (higher value are black and lower values are white)
colormap(colormap);

textStrings = num2str(TrainType3_TestType2(:), '%0.2f');      %Create strings from the matrix value
textStrings = strtrim(cellstr(textStrings));    %Remove any space padding
[x, y] = meshgrid(1:3);                         %Create x and y coordinates for the strings
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', 'FontSize', 20);   %Plot the strings
xlabel('Delay(Type2)', 'FontSize', 20)
ylabel('Threshold', 'FontSize', 20)

set(gca,'XTick',1:4,...                         %# Change the axes tick marks
        'XTickLabel',{'0','1','2'},...  %#   and tick labels
        'YTick',1:4,...
        'YTickLabel',{'1','2','3'},...
        'TickLength',[0 0]);
set(gca, 'FontSize', 20)

subplot(1,3,3)
imagesc(TrainType3_TestType3);            %Create a colored plot of the matrix value
%colormap(flipud(gray));     %Change the colormap to gray (higher value are black and lower values are white)
colormap(colormap);

textStrings = num2str(TrainType3_TestType3(:), '%0.2f');      %Create strings from the matrix value
textStrings = strtrim(cellstr(textStrings));    %Remove any space padding
[x, y] = meshgrid(1:3);                         %Create x and y coordinates for the strings
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', 'FontSize', 20);   %Plot the strings
xlabel('Delay(Type3)', 'FontSize', 20)
ylabel('Threshold', 'FontSize', 20)

set(gca,'XTick',1:4,...                         %# Change the axes tick marks
        'XTickLabel',{'0','1','2'},...  %#   and tick labels
        'YTick',1:4,...
        'YTickLabel',{'1','2','3'},...
        'TickLength',[0 0]);
set(gca, 'FontSize', 20)