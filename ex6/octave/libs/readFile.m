function file_contents = readFile(filename)
    fd = fopen(filename);
    if fd
        file_contents = fscanf(fd, '%c', inf);
        fclose(fd);
    else
        file_contents = '';
        fprintf('Unable to open %s\n', filename);
    end
end