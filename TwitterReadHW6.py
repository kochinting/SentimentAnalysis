__author__ = 'Ko, Chin-Ting'
import twitter
import simplejson as json


CONSUMER_KEY = 'DSZcbcVW6BrySHmeoBpRTUGLI'
CONSUMER_SECRET = 'SilPAHqD7OCYDseI7m1sPkG4JIIsudnP9pOAbzRen5y2n2yq00'
OAUTH_TOKEN= '783172770188320768-zFAMLsiOAkQ9lWlBLE3Q4tBK9UnhvoS'
OAUTH_TOKEN_SECRET = '1BSWG769aMIeKkHGlWlzsVqdokvlSGzRaf7pD8zfkRWJQ'

auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                           CONSUMER_KEY, CONSUMER_SECRET)
twitter_api = twitter.Twitter(auth=auth)

q= "Allegiant"
count = 100

search_results = twitter_api.search.tweets(q=q, count=count)
statuses = search_results['statuses']

for _ in range(100):
    print ("Length of statuses", len(statuses))
    try:
        next_results = search_results['search_metadata']['next_results']
    except KeyError:
        break

    kwargs = dict([kv.split ('=') for kv in next_results[1:].split("&")])
    search_results = twitter_api.search.tweets(**kwargs)
    statuses += search_results['statuses']

print (json.dumps(statuses[0], indent=1))

status_texts = [ status['text']
                    for status in statuses ]

screen_names = [ user_mention['screen_name']
                 for status in statuses
                    for user_mention in status['entities']['user_mentions']]
hashtags = [ hashtag['text']
             for status in statuses
                for hashtag in status['entities']['hashtags']]

words = [ w
          for t in status_texts
            for w in t.split()]


print (json.dumps(status_texts[0:100], indent=1))

file = open("twitterMovie.txt", "w")
file.write(json.dumps(status_texts[0:1000],indent=1))
file.close()