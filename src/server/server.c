#define _GNU_SOURCE

#include "server/server.h"
#include "server/routes.h"
#include "server/request_parse.h"
#include "util/log.h"

#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAX_CLIENTS    64
#define RECV_BUF_SIZE  (256 * 1024)  // 256KB per client

static volatile int g_running = 1;

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

typedef struct {
    int    fd;
    char  *buf;
    size_t len;
    size_t cap;
} ClientConn;

static int create_listen_socket(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        LOG_ERROR("server: socket() failed: %s", strerror(errno));
        return -1;
    }

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons((uint16_t)port),
        .sin_addr.s_addr = INADDR_ANY,
    };

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        LOG_ERROR("server: bind() failed on port %d: %s", port, strerror(errno));
        close(fd);
        return -1;
    }

    if (listen(fd, 128) < 0) {
        LOG_ERROR("server: listen() failed: %s", strerror(errno));
        close(fd);
        return -1;
    }

    return fd;
}

int server_run(Model *model, int port) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);

    int listen_fd = create_listen_socket(port);
    if (listen_fd < 0) return -1;

    LOG_INFO("server: listening on http://0.0.0.0:%d", port);
    LOG_INFO("server: endpoints: /v1/chat/completions, /v1/models, /health");

    ClientConn clients[MAX_CLIENTS] = {0};
    struct pollfd fds[MAX_CLIENTS + 1];

    // First pollfd is the listening socket
    fds[0].fd = listen_fd;
    fds[0].events = POLLIN;

    for (int i = 0; i < MAX_CLIENTS; i++) {
        clients[i].fd = -1;
        fds[i + 1].fd = -1;
        fds[i + 1].events = 0;
    }

    while (g_running) {
        int nfds = 1;
        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clients[i].fd >= 0) {
                fds[nfds].fd = clients[i].fd;
                fds[nfds].events = POLLIN;
                nfds++;
            }
        }

        int ready = poll(fds, (nfds_t)nfds, 1000); // 1s timeout
        if (ready < 0) {
            if (errno == EINTR) continue;
            LOG_ERROR("server: poll() failed: %s", strerror(errno));
            break;
        }

        // Check for new connections
        if (fds[0].revents & POLLIN) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            int client_fd = accept(listen_fd, (struct sockaddr *)&client_addr,
                                   &addr_len);
            if (client_fd >= 0) {
                // Find empty slot
                int slot = -1;
                for (int i = 0; i < MAX_CLIENTS; i++) {
                    if (clients[i].fd < 0) { slot = i; break; }
                }
                if (slot >= 0) {
                    clients[slot].fd = client_fd;
                    clients[slot].buf = malloc(RECV_BUF_SIZE);
                    clients[slot].len = 0;
                    clients[slot].cap = RECV_BUF_SIZE;
                    LOG_DEBUG("server: accepted connection (fd=%d, slot=%d)",
                              client_fd, slot);
                } else {
                    LOG_WARN("server: max clients reached, rejecting");
                    close(client_fd);
                }
            }
        }

        // Check existing connections
        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clients[i].fd < 0) continue;

            // Find this fd in the pollfd array
            bool readable = false;
            for (int j = 1; j < nfds; j++) {
                if (fds[j].fd == clients[i].fd && (fds[j].revents & POLLIN)) {
                    readable = true;
                    break;
                }
            }
            if (!readable) continue;

            ssize_t n = recv(clients[i].fd,
                             clients[i].buf + clients[i].len,
                             clients[i].cap - clients[i].len, 0);

            if (n <= 0) {
                // Connection closed or error
                close(clients[i].fd);
                free(clients[i].buf);
                clients[i].fd = -1;
                clients[i].buf = NULL;
                clients[i].len = 0;
                continue;
            }

            clients[i].len += (size_t)n;

            // Try to parse request
            HttpRequest req;
            int consumed = http_parse_request(&req, clients[i].buf,
                                               clients[i].len);
            if (consumed > 0) {
                // Handle request
                route_handle(clients[i].fd, &req, model);
                http_request_free(&req);

                // Shift remaining data
                if ((size_t)consumed < clients[i].len) {
                    memmove(clients[i].buf,
                            clients[i].buf + consumed,
                            clients[i].len - (size_t)consumed);
                    clients[i].len -= (size_t)consumed;
                } else {
                    clients[i].len = 0;
                }

                // Close if not keep-alive
                if (!req.keep_alive) {
                    close(clients[i].fd);
                    free(clients[i].buf);
                    clients[i].fd = -1;
                    clients[i].buf = NULL;
                    clients[i].len = 0;
                }
            } else if (consumed < 0) {
                // Parse error
                close(clients[i].fd);
                free(clients[i].buf);
                clients[i].fd = -1;
                clients[i].buf = NULL;
                clients[i].len = 0;
            }
            // consumed == 0: incomplete, wait for more data
        }
    }

    LOG_INFO("server: shutting down");

    // Cleanup
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clients[i].fd >= 0) {
            close(clients[i].fd);
            free(clients[i].buf);
        }
    }
    close(listen_fd);

    return 0;
}
