from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter




def webscrappingAndSplittingIntoChunks():
    print("*******Text File Loader ******************")
    txtLoader = TextLoader("../WebData.txt")
    textData = txtLoader.load()
    print(textData[0].metadata)
    print(textData[0].page_content)
    print("*************************")

    ##Splitting text
    recChrTxtSplitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n"," "], chunk_size=200,chunk_overlap=0)
    chunks = recChrTxtSplitter.split_text(textData[0].page_content)
    print(len(chunks))
    print(chunks)
    for chunk in chunks:
        print(len(chunk))
    print("************URL Loader*************")

    # urlLoader = UnstructuredURLLoader(urls=["https://www.moneycontrol.com/news/opinion/sips-a-way-to-keep-emotions-away-2501737.html",
    # "https://www.moneycontrol.com/news/business/economy/budget-2020-tough-balancing-act-for-finance-ministry-4855451.html"])
    # urlData = urlLoader.load()
    # print(len(urlData))
    # print("*************************")




if __name__ == "__main__":
    webscrappingAndSplittingIntoChunks()