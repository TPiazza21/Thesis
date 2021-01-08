clear
allfolders = dir('.');
allfolders(1:2) = [];
dirindex = [allfolders(:).isdir];
allfolders = allfolders(dirindex);
normalize_coordinates = 1; % z-scoring
% fprintf("%s", allfolders);
for i = 1:length(allfolders)
    file = dir(fullfile(allfolders(i).name, 'generatedata_*.m'));
    filename = [allfolders(i).name, '/', file.name];
    disp(filename)
    %fprintf("Well, this is the filename\n");
    %fprintf(filename);
    run(filename)
end
disp(i)