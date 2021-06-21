class Matcher:
    def __init__(self, _matcher):
        self.matcher = _matcher

    def match(self, kp1, kp2, des1, des2):
        matches_all = self.matcher.knnMatch(des1, des2, k=2)
        #matches_all = matcher.match(des1, des2)
        #matches_gms = matchGMS(shape1[:2], shape2[:2], kp1, kp2, matches_all)
        #matches_gms = sorted(matches_gms, key = lambda x: x.distance)
        good = []
        for m,n in matches_all:
            # 0.50 ???
            if m.distance < 0.50*n.distance and m.distance > 0.05*n.distance:
                good.append(m)
        return good


