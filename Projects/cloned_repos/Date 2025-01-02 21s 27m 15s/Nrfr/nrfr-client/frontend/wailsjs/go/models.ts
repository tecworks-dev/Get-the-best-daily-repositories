export namespace main {

    export class AppStatus {
        shizuku: boolean;
        // Go type: struct { Installed bool "json:\"installed\""; NeedUpdate bool "json:\"needUpdate\"" }
        nrfr: any;

        static createFrom(source: any = {}) {
            return new AppStatus(source);
        }

        constructor(source: any = {}) {
            if ('string' === typeof source) source = JSON.parse(source);
            this.shizuku = source["shizuku"];
            this.nrfr = this.convertValues(source["nrfr"], Object);
        }

        convertValues(a: any, classs: any, asMap: boolean = false): any {
            if (!a) {
                return a;
            }
            if (a.slice && a.map) {
                return (a as any[]).map(elem => this.convertValues(elem, classs));
            } else if ("object" === typeof a) {
                if (asMap) {
                    for (const key of Object.keys(a)) {
                        a[key] = new classs(a[key]);
                    }
                    return a;
                }
                return new classs(a);
            }
            return a;
        }
    }

    export class DeviceInfo {
        serial: string;
        state: string;
        product: string;
        model: string;

        static createFrom(source: any = {}) {
            return new DeviceInfo(source);
        }

        constructor(source: any = {}) {
            if ('string' === typeof source) source = JSON.parse(source);
            this.serial = source["serial"];
            this.state = source["state"];
            this.product = source["product"];
            this.model = source["model"];
        }
    }

}

