from numpy import int256
from CleaningScript import *
import plotly_express as px
import scipy

outStock, sales = beerClean()

oos_product, aggregated_sales = aggregateFuncs(outStock, sales)

oos_product, store_clusters = createStoreClusters(oos_product)
aggregated_sales, productClusters, clusterMap = createProductClusters(aggregated_sales)
oos_product['UPC Cluster'] = oos_product.UPC.astype(np.uint).map(clusterMap)

uniqueUPC = list(outStock.UPC.unique())
uniqueStore = list(outStock.StoreNum.unique())
weeks = [week for week in range(1,53)]

fullIndex = pd.MultiIndex.from_product(
    [
        weeks, 
        uniqueUPC, 
        uniqueStore
        ],
        names = ['Week', 'UPC', 'Store']
)
groupedOOS = groupedOOS.reindex(fullIndex, fill_value = 0)

oos_product.UPC.astype(np.uint)