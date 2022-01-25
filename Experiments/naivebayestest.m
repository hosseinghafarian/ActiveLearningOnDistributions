x1 = 5 * rand(100,1);
y1 = 5 * rand(100,1);
data1 = [x1,y1];
x2 = -5 * rand(100,1);
y2 =  5 * rand(100,1);
data2 = [x2,y2];
x3 = -5 * rand(100,1);
y3 = -5 * rand(100,1);
data3 = [x3,y3];
traindata = [data1(1:50,:);data2(1:50,:);data3(1:50,:)];
testdata = [data1(51:100,:);data2(51:100,:);data3(51:100,:)];
label = [repmat('x+y+',50,1);repmat('x-y+',50,1);repmat('x-y-',50,1)];

nb = NaiveBayes.fit(traindata, label);
ClassifierOut = predict(nb,testdata);
