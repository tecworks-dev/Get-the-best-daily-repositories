import { Skeleton } from "./ui/skeleton";

export interface TImage {
  id: string;
  status: string;
  imageUrl: string;
}

const DEFAULT_BLUR_IMAGE =
  "https://static.vecteezy.com/system/resources/thumbnails/016/894/217/small/white-background-white-polished-metal-abstract-white-gradient-background-blurred-white-backdrop-illustration-vector.jpg";

export function ImageCard(props: TImage) {
  return (
    <div className="border rounded-xl max-w-[400px] cursor-pointer">
      <div className="flex p-4 gap-4">
        {props.status === "Generated" ? (
          <img src={props.imageUrl} className="rounded" />
        ) : (
          <img src={DEFAULT_BLUR_IMAGE} />
        )}
      </div>
    </div>
  );
}

export function ImageCardSkeleton() {
  return (
    <div className="border rounded-xl max-w-[400px] p-2 cursor-pointer w-full">
      <div className="flex p-4 gap-4">
        <Skeleton className="rounded h-40 w-[300px]" />
      </div>
    </div>
  );
}
