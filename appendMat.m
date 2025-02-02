function appendMat(path1,path2)
    a = load(path1).data
    b = load(path2).data
    Fs = load(path1).Fs
    a = [a;b]
    save('left_trials','a','Fs')