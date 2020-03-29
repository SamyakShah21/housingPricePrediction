using CSV
using DataFrames
dataset = CSV.read("../data/housingPriceData.csv")
id = dataset.id;
price = dataset.price;
bedrooms = dataset.bedrooms;
bathrooms=dataset.bathrooms;
sqft_living=dataset.sqft_living;
len=length(id)
div=0.8*len

pricer=price[1:12969]
bedr=bedrooms[1:12969]
bathr=bathrooms[1:12969]
sqftr=sqft_living[1:12969]

pricev=price[12969:17290]
bedv=bedrooms[12969:17290]
bathv=bathrooms[12969:17290]
sqftv=sqft_living[12969:17290]

pricet=price[17290:21613]
bedt=bedrooms[17290:21613]
batht=bathrooms[17290:21613]
sqftt=sqft_living[17290:21613]

#normalize
function normalize(x)
    m=length(x)
    mean=sum(x)/m
    stdev=sqrt(sum((x.-mean).^2 )/m)
    y = (x.-mean)./stdev
    return y
end
function abs(x)
    if x<0
        return -x
    else
        return x
    end
end

p=length(bedr)
m=length(bedv) 
n=length(bedt)

x0=ones(p)
x1=normalize(bedr)
x2=normalize(bathr)
x3=normalize(sqftr)
x0v=ones(m)
x1v=normalize(bedv)
x2v=normalize(bathv)
x3v=normalize(sqftv)
x0t=ones(n)
x1t=normalize(bedt)
x2t=normalize(batht)
x3t=normalize(sqftt)

# # Define the features array
Xr = cat(x0, x1, x2,x3, dims=2)
Xv = cat(x0v, x1v, x2v,x3v, dims=2)
Xt= cat(x0t, x1t, x2t,x3t, dims=2)

# Get the variable we want to regress
Y =pricer
Yv=pricev
Yt=pricet

# Define a function to calculate cost function
function costFunction(X, Y, B ,lambd)
    m = length(Y)
    n= length(B)
    cost = sum(((X * B) - Y).^2)/(2*m) +  lambd*sum(abs.(B))/(2*m) - lambd*abs.(B[1])/(2*m)
    return cost
end

# # Initial coefficients
B = zeros(4, 1)
# Calcuate the cost with intial model parameters B=[0,0,0]
lambd=0
intialCost = costFunction(Xr, Y, B ,lambd)

function coordinateDescent(X, Y, B, lambd, learningRate, numIterations)
    cost=0
    m=length(Y)
    n=length(B)
    # do gradient descent for require number of iterations
    for iteration in 1:numIterations
        # Predict with current model B and find loss
        loss = (X * B) - Y
        # Compute Gradients: Ref to Andrew Ng. course notes linked on course page and Moodle
        for i=2:n
            rho=sum(X[(1:m),i].* (Y-(X*B) +(B[i].*X[(1:m),i])))/m
            
            if(rho > lambd/(2))
                B[i] = rho - (lambd/(2))
            elseif (rho < -1*lambd/(2))
                B[i] = rho + (lambd/(2))
            else
                B[i]=0
            end
        end
        B[1]=sum(X[(1:m),1] .* (Y - (X * B) + (B[1].*X[(1:m),1])))/m
    end
    return B, cost
end

#
learningRate = 0.001
newB, cost = coordinateDescent(Xr, Y, B,0, learningRate, 10000)

cost=zeros(4,1)
RSME=zeros(4,1)
lambdt = [10000,15000,16000,20000]
for i=1:10
    valB , cost[i] = coordinateDescent(Xr, Y, newB,lambdt[i], 0.001, 10000)
    pred = Xv* valB
    RSME[i]=sqrt(sum((Yv-pred).^2)/length(Yv))
end


ind=argmin(RSME)
finalB ,cost= coordinateDescent(Xr, Y, newB,lambdt[ind], 0.001, 10000)
YPred = Xt * finalB

RMSE= sqrt(sum((Yt-YPred).^2)/length(Yt))

ymean=sum(Yt)/length(Yt)
R2 =1-(sum((YPred-Yt).*(YPred-Yt))/sum((Yt.-ymean).*(Yt.-ymean)))
YPred = DataFrame(YPred)
output = CSV.write("../data/1a.csv", YPred)