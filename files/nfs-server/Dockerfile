FROM alpine:latest
# Optional build-time mirror (restricted networks). Default = upstream.
ARG APK_MIRROR=dl-cdn.alpinelinux.org
RUN sed -i "s|dl-cdn.alpinelinux.org|${APK_MIRROR}|g" /etc/apk/repositories \
 && apk add --no-cache nfs-utils bash
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 2049/tcp 111/tcp 111/udp
ENTRYPOINT ["/entrypoint.sh"]
