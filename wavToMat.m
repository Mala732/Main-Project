% Step 1: Read the .wav file
function wavToMat(filePath,savePath,no)
    [data, sampleRate] = audioread(strcat(filePath,int2str(no),'.wav'));  % Replace 'example.wav' with your .wav file name
    %data = bandpass(data,[3,50],sampleRate);
    
    % Step 2: Save to .mat file
    save(strcat(savePath, int2str(no),'.mat'), 'data', 'sampleRate');  % Replace 'example.mat' with your desired .mat file name
    
    disp('WAV file saved as MAT file successfully.');
end



