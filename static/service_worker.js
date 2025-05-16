self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open('v1').then(function(cache) {
      return cache.addAll([
        '/',  // 홈 경로만 캐싱
        // 불필요한 정적 파일 경로 제거
        '/static/icons/icon-192x192.png',
        '/static/icons/icon-512x512.png',
        '/static/manifest.json'
      ]);
    })
  );
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request).then(function(response) {
      return response || fetch(event.request);
    })
  );
});
