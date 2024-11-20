import renderItem from "@/app/renderItem";
import { FlashList } from "@shopify/flash-list";
import { useRef } from "react";
import { StyleSheet, View } from "react-native";

export default function HomeScreen() {
  const data = Array.from({ length: 1000 }, (_, i) => ({ id: i.toString() }));

  const scrollRef = useRef<FlashList<any>>(null);

  //   useEffect(() => {
  //     let amtPerInterval = 4;
  //     let index = amtPerInterval;
  //     const interval = setInterval(() => {
  //       scrollRef.current?.scrollToIndex({
  //         index,
  //       });
  //       index += amtPerInterval;
  //     }, 100);

  //     return () => clearInterval(interval);
  //   });

  return (
    <View style={[StyleSheet.absoluteFill, styles.outerContainer]}>
      <FlashList
        // style={[StyleSheet.absoluteFill, styles.scrollContainer]}
        data={data}
        renderItem={renderItem as any}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContainer}
        estimatedItemSize={389}
        // initialScrollIndex={500}
        ref={scrollRef}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  outerContainer: {
    backgroundColor: "#456",
  },
  scrollContainer: {
    // paddingHorizontal: 8,
  },
  titleContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  stepContainer: {
    gap: 8,
    marginBottom: 8,
  },
  reactLogo: {
    height: 178,
    width: 290,
    bottom: 0,
    left: 0,
    position: "absolute",
  },
  itemContainer: {
    // padding: 4,
    // borderBottomWidth: 1,
    // borderBottomColor: "#ccc",
  },
  listContainer: {
    paddingHorizontal: 16,
    paddingTop: 48,
  },
  itemTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 8,
    color: "#1a1a1a",
  },
  itemBody: {
    fontSize: 14,
    color: "#666666",
    lineHeight: 20,
    flex: 1,
  },
  itemFooter: {
    flexDirection: "row",
    justifyContent: "flex-start",
    gap: 16,
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: "#f0f0f0",
  },
  footerText: {
    fontSize: 14,
    color: "#888888",
  },
});
