function [vocabList, vocabListReversed] = getVocabList(filename)
    fd = fopen(filename);
    n = 1899;

    % For ease of implementation, we use a struct to map the strings => integers
    % In practice, you'll want to use some form of hashmap
    % vocabList = cell(n, 1);
    vocabulary = cell(n, 1);
    for i = 1:n
        fscanf(fd, '%d', 1);
        vocabulary(i) = fscanf(fd, '%s', 1);
        % vocabList{i} = fscanf(fd, '%s', 1);
    end
    fclose(fd);
    vocabList = containers.Map(vocabulary, 1:n);
    vocabListReversed = containers.Map(1:n, vocabulary);
end