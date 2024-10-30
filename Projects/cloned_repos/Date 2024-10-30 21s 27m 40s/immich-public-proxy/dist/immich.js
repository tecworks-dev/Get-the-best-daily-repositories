"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tslib_1 = require("tslib");
const types_1 = require("./types");
const dayjs_1 = tslib_1.__importDefault(require("dayjs"));
const index_1 = require("./index");
class Immich {
    /**
     * Make a request to Immich API. We're not using the SDK to limit
     * the possible attack surface of this app.
     */
    request(endpoint) {
        return tslib_1.__awaiter(this, void 0, void 0, function* () {
            const res = yield fetch(process.env.IMMICH_URL + '/api' + endpoint, {
                headers: {
                    'x-api-key': process.env.API_KEY || ''
                }
            });
            if (res.status === 200) {
                const contentType = res.headers.get('Content-Type') || '';
                if (contentType.includes('application/json')) {
                    return res.json();
                }
                else {
                    return res;
                }
            }
            else {
                (0, index_1.log)('Immich API status ' + res.status);
                console.log(yield res.text());
            }
        });
    }
    /**
     * Query Immich for the SharedLink metadata for a given key.
     * The key is what is returned in the URL when you create a share in Immich.
     */
    getShareByKey(key) {
        return tslib_1.__awaiter(this, void 0, void 0, function* () {
            const link = (yield this.request('/shared-links/me?key=' + encodeURIComponent(key)));
            if (link) {
                if (link.expiresAt && (0, dayjs_1.default)(link.expiresAt) < (0, dayjs_1.default)()) {
                    // This link has expired
                    (0, index_1.log)('Expired link ' + key);
                }
                else {
                    // Filter assets to exclude trashed assets
                    link.assets = link.assets.filter(asset => !asset.isTrashed);
                    link.assets.forEach(asset => { asset.key = key; });
                    return link;
                }
            }
        });
    }
    /**
     * Stream asset buffer data from Immich.
     *
     * For photos, you can request 'thumbnail' or 'original' size.
     * For videos, it is Immich's streaming quality, not the original quality.
     */
    getAssetBuffer(asset, size) {
        return tslib_1.__awaiter(this, void 0, void 0, function* () {
            switch (asset.type) {
                case types_1.AssetType.image:
                    size = size === types_1.ImageSize.thumbnail ? types_1.ImageSize.thumbnail : types_1.ImageSize.original;
                    return this.request('/assets/' + encodeURIComponent(asset.id) + '/' + size + '?key=' + encodeURIComponent(asset.key));
                case types_1.AssetType.video:
                    return this.request('/assets/' + encodeURIComponent(asset.id) + '/video/playback?key=' + encodeURIComponent(asset.key));
            }
        });
    }
    /**
     * Get the content-type of an Immich asset
     */
    getContentType(asset) {
        return tslib_1.__awaiter(this, void 0, void 0, function* () {
            const assetBuffer = yield this.getAssetBuffer(asset);
            return assetBuffer.headers.get('Content-Type');
        });
    }
    /**
     * Return the image data URL for a photo
     */
    photoUrl(key, id, size) {
        return `/photo/${key}/${id}` + (size ? `?size=${size}` : '');
    }
    /**
     * Return the video data URL for a video
     */
    videoUrl(key, id) {
        return `/video/${key}/${id}`;
    }
    /**
     * Check if a provided ID matches the Immich ID format
     */
    isId(id) {
        return !!id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
    }
    /**
     * Check if a provided key matches the Immich shared-link key format
     */
    isKey(key) {
        return !!key.match(/^[\w-]+$/);
    }
}
const immich = new Immich();
exports.default = immich;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW1taWNoLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vc3JjL2ltbWljaC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7QUFBQSxtQ0FBd0U7QUFDeEUsMERBQXlCO0FBQ3pCLG1DQUE2QjtBQUU3QixNQUFNLE1BQU07SUFDVjs7O09BR0c7SUFDRyxPQUFPLENBQUUsUUFBZ0I7O1lBQzdCLE1BQU0sR0FBRyxHQUFHLE1BQU0sS0FBSyxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsVUFBVSxHQUFHLE1BQU0sR0FBRyxRQUFRLEVBQUU7Z0JBQ2xFLE9BQU8sRUFBRTtvQkFDUCxXQUFXLEVBQUUsT0FBTyxDQUFDLEdBQUcsQ0FBQyxPQUFPLElBQUksRUFBRTtpQkFDdkM7YUFDRixDQUFDLENBQUE7WUFDRixJQUFJLEdBQUcsQ0FBQyxNQUFNLEtBQUssR0FBRyxFQUFFLENBQUM7Z0JBQ3ZCLE1BQU0sV0FBVyxHQUFHLEdBQUcsQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLGNBQWMsQ0FBQyxJQUFJLEVBQUUsQ0FBQTtnQkFDekQsSUFBSSxXQUFXLENBQUMsUUFBUSxDQUFDLGtCQUFrQixDQUFDLEVBQUUsQ0FBQztvQkFDN0MsT0FBTyxHQUFHLENBQUMsSUFBSSxFQUFFLENBQUE7Z0JBQ25CLENBQUM7cUJBQU0sQ0FBQztvQkFDTixPQUFPLEdBQUcsQ0FBQTtnQkFDWixDQUFDO1lBQ0gsQ0FBQztpQkFBTSxDQUFDO2dCQUNOLElBQUEsV0FBRyxFQUFDLG9CQUFvQixHQUFHLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQTtnQkFDdEMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFBO1lBQy9CLENBQUM7UUFDSCxDQUFDO0tBQUE7SUFFRDs7O09BR0c7SUFDRyxhQUFhLENBQUUsR0FBVzs7WUFDOUIsTUFBTSxJQUFJLEdBQUcsQ0FBQyxNQUFNLElBQUksQ0FBQyxPQUFPLENBQUMsdUJBQXVCLEdBQUcs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WyJpbXBvcnQgeyBBbGJ1bSwgQXNzZXQsIEFzc2V0VHlwZSwgSW1hZ2VTaXplLCBTaGFyZWRMaW5rIH0gZnJvbSAnLi90eXBlcydcbmltcG9ydCBkYXlqcyBmcm9tICdkYXlqcydcbmltcG9ydCB7IGxvZyB9IGZyb20gJy4vaW5kZXgnXG5cbmNsYXNzIEltbWljaCB7XG4gIC8qKlxuICAgKiBNYWtlIGEgcmVxdWVzdCB0byBJbW1pY2ggQVBJLiBXZSdyZSBub3QgdXNpbmcgdGhlIFNESyB0byBsaW1pdFxuICAgKiB0aGUgcG9zc2libGUgYXR0YWNrIHN1cmZhY2Ugb2YgdGhpcyBhcHAuXG4gICAqL1xuICBhc3luYyByZXF1ZXN0IChlbmRwb2ludDogc3RyaW5nKSB7XG4gICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2gocHJvY2Vzcy5lbnYuSU1NSUNIX1VSTCArICcvYXBpJyArIGVuZHBvaW50LCB7XG4gICAgICBoZWFkZXJzOiB7XG4gICAgICAgICd4LWFwaS1rZXknOiBwcm9jZXNzLmVudi5BUElfS0VZIHx8ICcnXG4gICAgICB9XG4gICAgfSlcbiAgICBpZiAocmVzLnN0YXR1cyA9PT0gMjAwKSB7XG4gICAgICBjb25zdCBjb250ZW50VHlwZSA9IHJlcy5oZWFkZXJzLmdldCgnQ29udGVudC1UeXBlJykgfHwgJydcbiAgICAgIGlmIChjb250ZW50VHlwZS5pbmNsdWRlcygnYXBwbGljYXRpb24vanNvbicpKSB7XG4gICAgICAgIHJldHVybiByZXMuanNvbigpXG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZXR1cm4gcmVzXG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIGxvZygnSW1taWNoIEFQSSBzdGF0dXMgJyArIHJlcy5zdGF0dXMpXG4gICAgICBjb25zb2xlLmxvZyhhd2FpdCByZXMudGV4dCgpKVxuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBRdWVyeSBJbW1pY2ggZm9yIHRoZSBTaGFyZWRMaW5rIG1ldGFkYXRhIGZvciBhIGdpdmVuIGtleS5cbiAgICogVGhlIGtleSBpcyB3aGF0IGlzIHJldHVybmVkIGluIHRoZSBVUkwgd2hlbiB5b3UgY3JlYXRlIGEgc2hhcmUgaW4gSW1taWNoLlxuICAgKi9cbiAgYXN5bmMgZ2V0U2hhcmVCeUtleSAoa2V5OiBzdHJpbmcpIHtcbiAgICBjb25zdCBsaW5rID0gKGF3YWl0IHRoaXMucmVxdWVzdCgnL3NoYXJlZC1saW5rcy9tZT9rZXk9JyArIGVuY29kZVVSSUNvbXBvbmVudChrZXkpKSkgYXMgU2hhcmVkTGlua1xuICAgIGlmIChsaW5rKSB7XG4gICAgICBpZiAobGluay5leHBpcmVzQXQgJiYgZGF5anMobGluay5leHBpcmVzQXQpIDwgZGF5anMoKSkge1xuICAgICAgICAvLyBUaGlzIGxpbmsgaGFzIGV4cGlyZWRcbiAgICAgICAgbG9nKCdFeHBpcmVkIGxpbmsgJyArIGtleSlcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIC8vIEZpbHRlciBhc3NldHMgdG8gZXhjbHVkZSB0cmFzaGVkIGFzc2V0c1xuICAgICAgICBsaW5rLmFzc2V0cyA9IGxpbmsuYXNzZXRzLmZpbHRlcihhc3NldCA9PiAhYXNzZXQuaXNUcmFzaGVkKVxuICAgICAgICBsaW5rLmFzc2V0cy5mb3JFYWNoKGFzc2V0ID0+IHsgYXNzZXQua2V5ID0ga2V5IH0pXG4gICAgICAgIHJldHVybiBsaW5rXG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIFN0cmVhbSBhc3NldCBidWZmZXIgZGF0YSBmcm9tIEltbWljaC5cbiAgICpcbiAgICogRm9yIHBob3RvcywgeW91IGNhbiByZXF1ZXN0ICd0aHVtYm5haWwnIG9yICdvcmlnaW5hbCcgc2l6ZS5cbiAgICogRm9yIHZpZGVvcywgaXQgaXMgSW1taWNoJ3Mgc3RyZWFtaW5nIHF1YWxpdHksIG5vdCB0aGUgb3JpZ2luYWwgcXVhbGl0eS5cbiAgICovXG4gIGFzeW5jIGdldEFzc2V0QnVmZmVyIChhc3NldDogQXNzZXQsIHNpemU/OiBJbWFnZVNpemUpIHtcbiAgICBzd2l0Y2ggKGFzc2V0LnR5cGUpIHtcbiAgICAgIGNhc2UgQXNzZXRUeXBlLmltYWdlOlxuICAgICAgICBzaXplID0gc2l6ZSA9PT0gSW1hZ2VTaXplLnRodW1ibmFpbCA/IEltYWdlU2l6ZS50aHVtYm5haWwgOiBJbWFnZVNpemUub3JpZ2luYWxcbiAgICAgICAgcmV0dXJuIHRoaXMucmVxdWVzdCgnL2Fzc2V0cy8nICsgZW5jb2RlVVJJQ29tcG9uZW50KGFzc2V0LmlkKSArICcvJyArIHNpemUgKyAnP2tleT0nICsgZW5jb2RlVVJJQ29tcG9uZW50KGFzc2V0LmtleSkpXG4gICAgICBjYXNlIEFzc2V0VHlwZS52aWRlbzpcbiAgICAgICAgcmV0dXJuIHRoaXMucmVxdWVzdCgnL2Fzc2V0cy8nICsgZW5jb2RlVVJJQ29tcG9uZW50KGFzc2V0LmlkKSArICcvdmlkZW8vcGxheWJhY2s/a2V5PScgKyBlbmNvZGVVUklDb21wb25lbnQoYXNzZXQua2V5KSlcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogR2V0IHRoZSBjb250ZW50LXR5cGUgb2YgYW4gSW1taWNoIGFzc2V0XG4gICAqL1xuICBhc3luYyBnZXRDb250ZW50VHlwZSAoYXNzZXQ6IEFzc2V0KSB7XG4gICAgY29uc3QgYXNzZXRCdWZmZXIgPSBhd2FpdCB0aGlzLmdldEFzc2V0QnVmZmVyKGFzc2V0KVxuICAgIHJldHVybiBhc3NldEJ1ZmZlci5oZWFkZXJzLmdldCgnQ29udGVudC1UeXBlJylcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXR1cm4gdGhlIGltYWdlIGRhdGEgVVJMIGZvciBhIHBob3RvXG4gICAqL1xuICBwaG90b1VybCAoa2V5OiBzdHJpbmcsIGlkOiBzdHJpbmcsIHNpemU/OiBJbWFnZVNpemUpIHtcbiAgICByZXR1cm4gYC9waG90by8ke2tleX0vJHtpZH1gICsgKHNpemUgPyBgP3NpemU9JHtzaXplfWAgOiAnJylcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXR1cm4gdGhlIHZpZGVvIGRhdGEgVVJMIGZvciBhIHZpZGVvXG4gICAqL1xuICB2aWRlb1VybCAoa2V5OiBzdHJpbmcsIGlkOiBzdHJpbmcpIHtcbiAgICByZXR1cm4gYC92aWRlby8ke2tleX0vJHtpZH1gXG4gIH1cblxuICAvKipcbiAgICogQ2hlY2sgaWYgYSBwcm92aWRlZCBJRCBtYXRjaGVzIHRoZSBJbW1pY2ggSUQgZm9ybWF0XG4gICAqL1xuICBpc0lkIChpZDogc3RyaW5nKSB7XG4gICAgcmV0dXJuICEhaWQubWF0Y2goL15bMC05YS1mXXs4fS1bMC05YS1mXXs0fS1bMC05YS1mXXs0fS1bMC05YS1mXXs0fS1bMC05YS1mXXsxMn0kLylcbiAgfVxuXG4gIC8qKlxuICAgKiBDaGVjayBpZiBhIHByb3ZpZGVkIGtleSBtYXRjaGVzIHRoZSBJbW1pY2ggc2hhcmVkLWxpbmsga2V5IGZvcm1hdFxuICAgKi9cbiAgaXNLZXkgKGtleTogc3RyaW5nKSB7XG4gICAgcmV0dXJuICEha2V5Lm1hdGNoKC9eW1xcdy1dKyQvKVxuICB9XG59XG5cbmNvbnN0IGltbWljaCA9IG5ldyBJbW1pY2goKVxuXG5leHBvcnQgZGVmYXVsdCBpbW1pY2hcbiJdfQ==