import pymongo

# Private Key: 69886c33-6cba-4ad0-bc2e-09bbcd3eb28f

client = pymongo.MongoClient('mongodb+srv://tobyd:root@fmp-o7ona.gcp.mongodb.net/test?retryWrites=true')


db = client['FMP-JSON']
col = db['Videos']


x = col.delete_many({})
