using CSV
using DataFrames
dataset = CSV.read("../data/housingPriceData.csv")
id = dataset.id;
price = dataset.price;
bedrooms = dataset.bedrooms;
bathrooms=dataset.bathrooms;
sqft_living=dataset.sqft_living;
len=length(id)

pricer=price[1:17290]
bedr=bedrooms[1:17290]
bathr=bathrooms[1:17290]
sqftr=sqft_living[1:17290]
pricet=price[17290:21613]
bedt=bedrooms[17290:21613]
batht=bathrooms[17290:21613]
sqftt=sqft_living[17290:21613]

#standardize
function standardize(x)
    m=length(x)
    mean=sum(x)/m
    stdev=sqrt(sum((x.-mean).^2 )/m)
    y = (x.-mean)./stdev
    return y
end

m=length(bedr)
n=length(bedt)

x1_2=bedr.*bedr
x2_2=sqftr.*sqftr
x1_x2=bedr.*sqftr
x1_2t=bedt.*bedt
x2_2t=sqftt.*sqftt
x1_x2t=bedt.*sqftt

x0=ones(m)
x1=standardize(bedr)
x2=standardize(sqftr)
x3=standardize(x1_2)
x4=standardize(x2_2)
x5=standardize(x1_x2)
x0t=ones(n)
x1t=standardize(bedt)
x2t=standardize(sqftt)
x3t=standardize(x1_2t)
x4t=standardize(x2_2t)
x5t=standardize(x1_x2t)

Xr = cat(x0,x1,x2,x3,x4,x5, dims=2)
Xt= cat(x0t,x1t,x2t,x3t,x4t,x5t, dims=2)

Y =pricer
Yt=pricet


function costFunction(X, Y, B)
    m = length(Y)
    cost = sum(((X * B) - Y).^2)/(2*m)
    return cost
end


B = zeros(6, 1)

intialCost = costFunction(Xr, Y, B)

function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)

    for iteration in 1:numIterations

        loss = (X * B) - Y

        gradient = (X' * loss)/m

        B = B - learningRate * gradient

        cost = costFunction(X, Y, B)

        costHistory[iteration] = cost
    end
    return B, costHistory
end

learningRate = 0.0001
newB, costHistory = gradientDescent(Xr, Y, B, learningRate, 100000)

YPred = Xt * newB

RMSE= sqrt(sum((Yt-YPred).^2)/length(Yt))

ymean=sum(Yt)/length(Yt)


R2 =1-(sum((YPred-Yt).*(YPred-Yt))/sum((Yt.-ymean).*(Yt.-ymean)))
YPred = DataFrame(YPred)
output = CSV.write("../data/1a.csv", YPred)
