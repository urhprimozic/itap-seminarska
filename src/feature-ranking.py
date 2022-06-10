if __name__ == "__main__":
    train = [2, 3, 4,  7, 8, 9,  12, 13, 14,  17, 19]
    # train = [2, 4,  8,  19]
    test = [1, 6, 11, 16]
    X_train = np.concatenate([get_data(i) for i in train])
    y_train = np.concatenate([get_reference(i) for i in train])
    X_test = np.concatenate([get_data(i) for i in test])
    y_test = np.concatenate([get_reference(i) for i in test])

    model = pixel_classifier(X_train, y_train, param_grid={'min_samples_split': [2, 4, 8],
                                                           'min_samples_leaf': [1, 2, 4],
                                                           'ccp_alpha': [0.0, 0.1, 0.2, 0.5]
                                                           })
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    axs[0].imshow(data_to_4D(get_data(1))[..., 8, [3, 2, 1]] * 3.5)
    axs[1].imshow(reshape_reference(model.predict(get_data(1))),
                  cmap=category_cmap, norm=category_norm)
    axs[2].imshow(reshape_reference(get_reference(1)),
                  cmap=category_cmap, norm=category_norm)

    for ax, title in zip(axs, ("Slika", "Napoved", "Referenca")):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")