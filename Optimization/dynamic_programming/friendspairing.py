def countFriendsPairings(n):
    dp = [0 for i in range(n + 1)]

    # Filling dp[] in bottom-up manner using
    # recursive formula explained above.
    for i in range(n + 1):

        if i <= 2:
            dp[i] = i
        else:
            dp[i] = dp[i - 1] + (i - 1) * dp[i - 2]

    return dp[n]


if __name__ == '__main__':
    print(countFriendsPairings(8))
