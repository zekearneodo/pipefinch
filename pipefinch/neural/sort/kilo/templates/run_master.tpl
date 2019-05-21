% run the master.m file in a try-catch mode, with all figures off, then exit
try
    set(gcf,'Visible','off');
    set(0,'DefaultFigureVisible','off');
    slCharacterEncoding('US-ASCII')
    fpath    = '${data_dir}';
    if ~exist(fpath, 'dir'); mkdir(fpath); end
    fprintf('Will run master.m in %s\n', fpath)
    master
    fprintf('Done running master.m, will exit with status 0. That is good\n');
    fprintf('return_value=0');
    exit(0)
catch ME
    % print whatever is known about the error and exit
    rethrow(ME)
    fprintf('Will exit with status -1. That is bad\n')
    fprintf('return_value=-1');
    exit(-1)
end