function splitData(file_path,no,div)
    a = load(file_path).data; 
    %[a,fs] = audioread(file_path,10000);
    %fs = load(file_path).sampleRate;
    fs = 10000;
    figure
    plot(a)
    len = length(a);
    step = int64(len/(div+1));
    sz = 10319%10084;%20167  ;%int64(40335/2)-1;%11295;
    data = [];
    cnt = 1
    figure
    for i = 1 : step : len-step
        mat = a(i:i+step);
        [~, ind] = max(mat);
        disp(ind)
        data= [data;a(i+ind-sz:i+ind+sz)'];
        subplot(5,9,cnt)
        plot(a(i+ind-sz:i+ind+sz));
        cnt = cnt+1;
    end
    file_name = "New folder/Dataset-7/left/left-1_"+int2str(no)+".mat";
    save(file_name,'data','fs');
end


