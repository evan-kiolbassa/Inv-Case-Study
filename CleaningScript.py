import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def year_month_wkExtract(series):

    '''
    A function that extracts features from the
    YEAR_MONTH_WK column using regex
    '''
    split_frame = series.str.split(' ', expand = True).copy()
    feature_frame = pd.DataFrame()
    feature_frame['Week'] = split_frame[8].str.replace('\D', '',
         regex = True)
    feature_frame['Year'] = split_frame[0]
    feature_frame['Period'] = split_frame[3]

    return feature_frame

def beerClean():

    outStock = pd.read_csv('F:/InBev/out_of_stock_data.csv')
    sales = pd.read_csv('F:/InBev/sales_data_weekly.csv')

    outStock_feature = year_month_wkExtract(outStock['YEAR_MONTH_WK'])
    sales_features = year_month_wkExtract(sales['YEAR_MONTH_WK'])

    outStock.loc[:,
    list(outStock_feature.columns)] = outStock_feature

    sales.loc[:,
    list(sales_features.columns)] = sales_features

    return outStock, sales



def aggregateFuncs(outStock, sales):

    oos_product = outStock.groupby(
        [
            'UPC',
            'Week',
            'DivisionNum',
            'DSD_Vendor_Name',
            'StoreNum'
    ]
    )['DATE_KEY'\
    ].count().reset_index().rename(columns = {'DATE_KEY': 'Out of Stock Count'})
    
    aggregated_sales = sales.groupby(
        [
            'UPC', 
            'Week', 
            'RE_STO_NUM', 
            'RPT_SHORT_DESC']
        ).agg(
            {
        'SCANNED_RETAIL_DOLLARS' : 'sum',
        'SCANNED_MOVEMENT' : 'sum',
        'SCANNED_LBS' : 'sum',
        'GROSS_MARGIN_DOLLARS' : 'sum'
    }
    )
    aggregated_sales = aggregated_sales.reset_index()
    '''
    Price per unit sold is a feature that details the unit price of any UPC.
    '''
    aggregated_sales['Price per Unit Sold'] = aggregated_sales.SCANNED_RETAIL_DOLLARS / aggregated_sales.SCANNED_MOVEMENT
    aggregated_sales = aggregated_sales.dropna()
    avgUnitPrice = aggregated_sales.groupby(['UPC', 
    'Week'])['Price per Unit Sold'].median().reset_index().rename(columns = {
        'Price per Unit Sold' : 'Average Price'
        })
    return oos_product, aggregated_sales

def createStoreClusters(oos_product):

    oos_pivot = pd.pivot_table(oos_product, 
    values = 'Out of Stock Count', 
    columns = 'Week', 
    index = 'StoreNum', aggfunc = lambda x: np.sum(x)).fillna(0)

    standardPivot = oos_pivot.apply(lambda x: (x - x.mean()) / x.std())
    
    clusters = []
    for i in range(1,20):
        km = KMeans(n_clusters=i)
        km.fit_predict(standardPivot)
        clusters.append(km.inertia_)

    clusters = pd.Series(clusters)
    clustNum = pd.Series([i for i in range(1,20)])

    km = KMeans(n_clusters=4)

    store_clusters = pd.Series(km.fit_predict(standardPivot), 
    index = standardPivot.index)

    clusterMap = {}
    for row in pd.DataFrame(store_clusters).itertuples():
        clusterMap[row[0]] = row[1]

    oos_product['Cluster'] = oos_product['StoreNum'].map(clusterMap)
    return oos_product, store_clusters

def createProductClusters(aggregated_sales):
    sales_pivot = pd.pivot_table(aggregated_sales, 
    values = 'SCANNED_RETAIL_DOLLARS', 
    columns = 'Week', 
    index = 'UPC', aggfunc = lambda x: np.sum(x)).fillna(0)

    standardPivot = sales_pivot.apply(lambda x: (x - x.mean()) / x.std())
    aggregated_sales.UPC = aggregated_sales.UPC.astype(np.uint)
    clusters = []
    for i in range(1,20):
        km = KMeans(n_clusters=i)
        km.fit_predict(standardPivot)
        clusters.append(km.inertia_)

    clusters = pd.Series(clusters)
    clustNum = pd.Series([i for i in range(1,20)])

    km = KMeans(n_clusters=4)

    productClusters = pd.Series(km.fit_predict(standardPivot), 
    index = standardPivot.index)

    clusterMap = {}
    for row in pd.DataFrame(productClusters).itertuples():
        clusterMap[row[0]] = row[1]

    aggregated_sales['Cluster'] = aggregated_sales['UPC'].map(clusterMap)
    return aggregated_sales, productClusters, clusterMap

def divisionNumAgg(oos_product):
    
    '''
    Aggregation of out-of-stock occurrences by unique region identifier.
    '''
    regions = oos_product.groupby(['Cluster', 'DivisionNum'])['Week'].count().reset_index()
    regions.Cluster = regions.Cluster.astype(str)
    regions.DivisionNum = regions.DivisionNum.astype(str)
    regions.columns = ['Cluster', 'DivisionNum', '# of Obs']
    return regions

def divisionHashing(regions):
    divisionObservations = regions.groupby('DivisionNum')['# of Obs'].sum().reset_index()
    divisionHash = {}
    for row in divisionObservations.itertuples(index = False):
        divisionHash[row[0]] = row[1]
    return divisionHash


def normalizeDivisionPop(regions):
    '''
    Generating a feature that normalizes the aggregated 
    out of stock counts by division into out of stock product 
    per 5000 occurrences
    '''
    divisionHash = divisionHashing(regions)
    mapHash = regions.DivisionNum.map(divisionHash)
    normalizedDiv = (regions['# of Obs'] / mapHash) * 5000
    return normalizedDiv

def timeSeriesFormatting():
    
    outStock = pd.read_csv('F:/InBev/out_of_stock_data.csv')
    sales = pd.read_csv('F:/InBev/sales_data_weekly.csv')
    
    outStock = outStock.dropna()
    '''
    Formatting of dates are not consistent across the DATE_KEY 
    columns. 
    '''
    slashIndexes = outStock[
        outStock.DATE_KEY.str.contains(
        '/', 
        regex = False
        )
        ].index
    
    splitDates = outStock.loc[
    slashIndexes,
    'DATE_KEY'
    ].str.split('/')


    slashStandardized = pd.Series([
        f'{date[2]}-0{date[0]}-0{date[1]}'  if (len(date[0])) == 1 & (len(date[1]) == 1)
        else f'{date[2]}-0{date[0]}-{date[1]}' if (len(date[0]) == 1) & (len(date[1]) == 2)
        else f'{date[2]}-{date[0]}-0{date[1]}' if (len(date[0])) == 2 & (len(date[1]) == 1)
        else f'{date[2]}-{date[0]}-{date[1]}' for date in splitDates
    ], index = slashIndexes)
    
   
    dashIndexes = outStock[
        outStock.DATE_KEY.str.contains(
        '-', 
        regex = False
        )
        ].index
    split1Dates = outStock.loc[
    dashIndexes,
    'DATE_KEY'
    ].str.split('-')   

    monthHash = {
        'month' : [
            f'0{month}' if month < 10
            else str(month) for month in range(1,13)
        ]
    }

    dashStandardized = pd.Series([
        f'{date[0]}-{date[2]}-{date[1]}' if str(date[1]) not in monthHash['month']
        else f'{date[0]}-{date[1]}-{date[2]}' for date in split1Dates
    ], index = dashIndexes)

    outStock.loc[
        slashIndexes, 
        'DATE_KEY'
        ] = slashStandardized

    outStock.loc[
        dashIndexes, 
        'DATE_KEY'
        ] = dashStandardized

    outStock.loc[:, [
    'Week',
    'Year',
    'Period'
    ]] = year_month_wkExtract(outStock.YEAR_MONTH_WK)

    
    return outStock, sales

def dateImputation(outStock):
    '''
    88633 rows have integers as dates in the DATE_KEY column.
    They all occurr in period 7 week 28. A random sampling technique 
    from numpy can be used for imputation with the constraints of
    an array of values from period 7 week 28 that excludes the invalid 
    date values.
    '''
    invalidDates = [
    '44055',
    '44056',
    '44057',
    '44058',
    '44054',
    '44053',
    '44052'
    ]
    
    invalidIndex = outStock[outStock.DATE_KEY.isin(invalidDates)].index

    validDates = list(outStock[
    (outStock.Period == '07') & (outStock.Week == '28') & ~outStock.DATE_KEY.isin(invalidDates)
    ].DATE_KEY.unique())

    imputedDates = pd.Series(np.random.choice(validDates, 
    size = 88633), 
    index = invalidIndex)

    outStock.loc[
        invalidIndex, 
        'DATE_KEY'
        ] = imputedDates

    outStock.DATE_KEY = pd.to_datetime(outStock.DATE_KEY)

    outStock.groupby([pd.Grouper(freq = 'W'), 
    'DivisionNum' ,
    'StoreNum', 
    'UPC'
    ])['Period'].count().reset_index().rename(
        columns = {
        'Period' : 'Out of Stock Count'
        }
        ).set_index('DATE_KEY')

    groupedOOS = outStock.groupby(
    [
        pd.Grouper(freq = 'W', dropna = False), 
        'Week',
        'UPC',
        'StoreNum'
    ]
    )['Period'].count().reset_index().rename(
        columns= {
        'Period' : 'Out of Stock Count'
        }
        )
    return outStock


